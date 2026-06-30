"""GasVision AI support backend.

This service provides a production-style backend for the customer support web app.
It intentionally separates:
- QR access validation
- FAQ flow (no LLM)
- dispatcher contact flow (no LLM)
- AI chat flow via LangGraph + DeepSeek
- escalation delivery to event-service
- audit logging to PostgreSQL

The code includes extensive comments because it is intended to be reused in
project documentation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


DATABASE_URL = require_env("DATABASE_URL")
EVENT_SERVICE_BASE_URL = require_env("EVENT_SERVICE_BASE_URL").rstrip("/")
DISPATCHER_PHONE = require_env("DISPATCHER_PHONE")
QR_ACCESS_SECRET = require_env("QR_ACCESS_SECRET")
QR_TOKEN_MAX_AGE_SECONDS = int(os.getenv("QR_TOKEN_MAX_AGE_SECONDS", "86400"))
DEEPSEEK_API_KEY = require_env("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "90"))
DEEPSEEK_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "2"))
CORS_ORIGINS = [item.strip() for item in require_env("CORS_ORIGINS").split(",") if item.strip()]
YANDEX_SPEECHKIT_API_KEY = os.getenv("YANDEX_SPEECHKIT_API_KEY", "")
YANDEX_SPEECHKIT_FOLDER_ID = os.getenv("YANDEX_SPEECHKIT_FOLDER_ID", "")
YANDEX_SPEECHKIT_LANG = os.getenv("YANDEX_SPEECHKIT_LANG", "ru-RU")
YANDEX_SPEECHKIT_TOPIC = os.getenv("YANDEX_SPEECHKIT_TOPIC", "general")
YANDEX_SPEECHKIT_TIMEOUT_SECONDS = float(os.getenv("YANDEX_SPEECHKIT_TIMEOUT_SECONDS", "30"))
VOICE_MAX_BYTES = int(os.getenv("VOICE_MAX_BYTES", str(3 * 1024 * 1024)))


# ---------------------------------------------------------------------------
# Static station map for demo / MVP.
# In production this can be replaced with a DB table or station service.
# ---------------------------------------------------------------------------
STATIONS = {
    "AZS-001": {"station_code": "AZS-001", "station_name": "АЗС #101"},
    "AZS-002": {"station_code": "AZS-002", "station_name": "АЗС 002"},
}


FAQ_ITEMS = [
    {
        "id": "payment",
        "question": "Как провести оплату?",
        "answer": "Для оплаты выберите номер колонки на терминале, укажите сумму или объем, подтвердите оплату и дождитесь сообщения об успешной транзакции. После этого можно начинать заправку.",
    },
    {
        "id": "receipt",
        "question": "Как получить чек?",
        "answer": "После завершения оплаты чек можно получить на терминале. Если поддерживается электронный чек, следуйте инструкции на экране.",
    },
    {
        "id": "cancel",
        "question": "Как отменить оплату?",
        "answer": "Если оплата еще не подтверждена, отмените операцию на терминале. Если средства уже списаны, обратитесь к диспетчеру.",
    },
]


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class ConversationMessage(Base):
    """Audit log of all user-visible interactions.

    We intentionally keep the schema very simple:
    - session_id groups one temporary visit from one QR-entry flow
    - station_code tells which station was encoded in QR
    - source shows how the message was generated
    - role indicates whether it is a user or assistant message
    """

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    station_code: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# QR access utilities
# ---------------------------------------------------------------------------
def get_token_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(QR_ACCESS_SECRET, salt="gasvision-qr")


@dataclass
class QRAccess:
    station_code: str
    station_name: str


def validate_qr_token(access_token: str) -> QRAccess:
    """Validate signed QR token and extract station information.

    Access is allowed only when a valid signed token exists in the URL.
    This directly implements the product requirement that the site is entered
    only via QR-code.
    """

    serializer = get_token_serializer()
    try:
        payload = serializer.loads(access_token, max_age=QR_TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired as exc:
        raise HTTPException(status_code=401, detail="QR code expired") from exc
    except BadSignature as exc:
        raise HTTPException(status_code=401, detail="Invalid QR token") from exc

    station_code = payload.get("station_code")
    station = STATIONS.get(station_code)
    if station is None:
        raise HTTPException(status_code=404, detail="Unknown station")

    return QRAccess(station_code=station["station_code"], station_name=station["station_name"])


# ---------------------------------------------------------------------------
# Event-service integration
# ---------------------------------------------------------------------------
class EventServiceClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    async def create_escalation(self, title: str, station_code: str) -> bool:
        """Create escalation in event-service.

        We keep payload shape compatible with the existing event-service used in
        the rest of GasVision.
        """

        payload = {
            "source": "ai",
            "title": title,
            "station_code": station_code,
            "camera_code": None,
            "severity": "med",
            "status": "open",
            "media": [],
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{self._base_url}/v1/events", json=payload)
                response.raise_for_status()
            return True
        except Exception:
            # For demo robustness we do not crash the request when event-service
            # is temporarily unavailable.
            return False


event_client = EventServiceClient(EVENT_SERVICE_BASE_URL)


# ---------------------------------------------------------------------------
# Speech-to-text integration
# ---------------------------------------------------------------------------
class SpeechKitClient:
    """Small wrapper around Yandex SpeechKit synchronous STT.

    Browser MediaRecorder usually produces Ogg/Opus or WebM/Opus. SpeechKit
    accepts OggOpus, so the backend normalizes uploads with ffmpeg before
    sending them to Yandex.
    """

    _recognize_url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

    def __init__(self, api_key: str, folder_id: str = "") -> None:
        self._api_key = api_key
        self._folder_id = folder_id

    def _ensure_configured(self) -> None:
        if not self._api_key:
            raise HTTPException(status_code=503, detail="Speech-to-text is not configured")

    @staticmethod
    def _convert_to_oggopus(audio: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=True) as source:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as target:
                source.write(audio)
                source.flush()
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    source.name,
                    "-vn",
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    target.name,
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, timeout=20)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    raise HTTPException(status_code=400, detail="Unsupported or invalid audio") from exc

                target.seek(0)
                return target.read()

    async def transcribe(self, audio: bytes) -> str:
        self._ensure_configured()
        ogg_audio = self._convert_to_oggopus(audio)
        params = {
            "lang": YANDEX_SPEECHKIT_LANG,
            "topic": YANDEX_SPEECHKIT_TOPIC,
            "format": "oggopus",
        }
        if self._folder_id:
            params["folderId"] = self._folder_id

        headers = {"Authorization": f"Api-Key {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=YANDEX_SPEECHKIT_TIMEOUT_SECONDS) as client:
                response = await client.post(self._recognize_url, params=params, content=ogg_audio, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail="SpeechKit rejected the audio") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="SpeechKit request failed") from exc

        return str(response.json().get("result", "")).strip()


speechkit_client = SpeechKitClient(YANDEX_SPEECHKIT_API_KEY, YANDEX_SPEECHKIT_FOLDER_ID)


# ---------------------------------------------------------------------------
# LangGraph + DeepSeek setup
# ---------------------------------------------------------------------------
class ScenarioId(str, Enum):
    FUEL_CASH = "fuel_cash"
    FUEL_CASH_DISCOUNT = "fuel_cash_discount"
    FUEL_FUEL_CARD = "fuel_fuel_card"
    FUEL_BANK_CARD = "fuel_bank_card"
    REFUND_CASH = "refund_cash"
    REFUND_CARD = "refund_card"
    WRONG_PARKING = "wrong_parking"
    WRONG_FUEL = "wrong_fuel"
    WRONG_PAYMENT_METHOD = "wrong_payment_method"
    DISCOUNT_CARD_ORIENTATION = "discount_card_orientation"
    FUEL_CARD_ORIENTATION = "fuel_card_orientation"
    BANK_CARD_ORIENTATION = "bank_card_orientation"
    FUEL_NOT_FIT = "fuel_not_fit"
    FORBIDDEN_CONTAINER = "forbidden_container"
    FORGOT_CARD = "forgot_card"
    FORGOT_NOZZLE = "forgot_nozzle"
    TERMINAL_DISCOUNT_FAILURE = "terminal_discount_failure"
    TERMINAL_CARD_OR_RECEIPT_FAILURE = "terminal_card_or_receipt_failure"
    GENERAL_REFUELING = "general_refueling"
    ENGINE_OFF_REQUIRED = "engine_off_required"
    STAY_NEAR_DISPENSER = "stay_near_dispenser"
    GENERAL_PAYMENT = "general_payment"
    RECEIPT_GENERAL = "receipt_general"
    NO_FUEL_FLOW = "no_fuel_flow"
    SAFETY_PROHIBITIONS = "safety_prohibitions"
    DISPATCHER_CONTACT_HELP = "dispatcher_contact_help"
    EMERGENCY_OR_UNSAFE = "emergency_or_unsafe"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SupportScenario:
    title: str
    answer: str
    classifier_notes: str


SCENARIOS: dict[ScenarioId, SupportScenario] = {
    ScenarioId.FUEL_CASH: SupportScenario(
        title="Заправка за наличный расчет без дисконтной карты",
        classifier_notes="Клиент спрашивает, как заправиться или оплатить наличными без дисконтной карты.",
        answer=(
            "Для заправки за наличный расчет: заглушите двигатель, установите пистолет в бензобак, "
            "на терминале нажмите «Наличные», затем «Оплата». Выберите нужный вид топлива, внесите купюры "
            "в купюроприемник и нажмите «Оплатить» для подтверждения. После заправки верните пистолет "
            "в исходное положение и заберите чек."
        ),
    ),
    ScenarioId.FUEL_CASH_DISCOUNT: SupportScenario(
        title="Заправка наличными с дисконтной картой",
        classifier_notes="Клиент платит наличными и хочет применить дисконтную карту.",
        answer=(
            "Для оплаты наличными с дисконтной картой: заглушите двигатель и установите пистолет в бензобак. "
            "Вставьте дисконтную карту в картоприемник чипом вверх и вперед до звукового сигнала, затем нажмите "
            "«Оплата». Выберите вид топлива, внесите купюры в купюроприемник и нажмите «Оплатить». После заправки "
            "верните пистолет, заберите дисконтную карту и чеки."
        ),
    ),
    ScenarioId.FUEL_FUEL_CARD: SupportScenario(
        title="Заправка по топливной карте",
        classifier_notes="Клиент хочет оплатить топливной картой или спрашивает порядок работы с ТК.",
        answer=(
            "Для заправки по топливной карте: заглушите двигатель, установите пистолет в бензобак и вставьте "
            "топливную карту в картоприемник чипом вверх и вперед до звукового сигнала. Нажмите «Оплата», выберите "
            "вид топлива, при запросе введите ПИН-код карты, нажмите «Набрать литры», затем «Выполнить» для "
            "подтверждения. После заправки верните пистолет, заберите топливную карту и чек."
        ),
    ),
    ScenarioId.FUEL_BANK_CARD: SupportScenario(
        title="Заправка по банковской карте",
        classifier_notes="Клиент хочет оплатить банковской картой или спрашивает порядок работы с БК.",
        answer=(
            "Для оплаты банковской картой: заглушите двигатель, установите пистолет в бензобак и вставьте карту "
            "в картоприемник магнитной лентой вниз по правому краю карты. Нажмите «Оплата», выберите вид топлива, "
            "при запросе введите ПИН-код, нажмите «Набрать литры», затем «Выполнить» для подтверждения. После "
            "заправки верните пистолет, заберите банковскую карту и чек."
        ),
    ),
    ScenarioId.REFUND_CASH: SupportScenario(
        title="Возврат денежных средств при оплате наличными",
        classifier_notes="Клиент спрашивает о возврате наличных, чеке возврата или не вошедшем в бак топливе после оплаты наличными.",
        answer=(
            "Для возврата при оплате наличными сначала верните пистолет в исходное положение. На терминале выберите "
            "«Наличные», затем «Возврат», введите код чека продажи, который указан внизу слева на чеке, и нажмите "
            "«Подтвердить» — терминал напечатает чек возврата. Для получения денег потребуется обратиться по адресу, "
            "который подсказывает диспетчер, и иметь при себе документ, удостоверяющий личность, чек продажи и чек возврата."
        ),
    ),
    ScenarioId.REFUND_CARD: SupportScenario(
        title="Возврат по топливной или банковской карте",
        classifier_notes="Клиент спрашивает о возврате остатка по банковской или топливной карте, если оплаченный объем не вошел.",
        answer=(
            "Если оплата была по банковской карте, сумма за топливо, которое не вошло в бак, автоматически возвращается "
            "на карту клиента. Если оплата была по топливной карте, верните пистолет, вставьте топливную карту чипом вверх "
            "и вперед до звукового сигнала и нажмите «Возврат» — чек возврата будет сформирован с новым остатком, а "
            "неиспользованная сумма возвращается на счет этой карты."
        ),
    ),
    ScenarioId.WRONG_PARKING: SupportScenario(
        title="Автомобиль подъехал не той стороной к колонке",
        classifier_notes="Клиент спрашивает, что делать, если горловина бака не со стороны ТРК или пистолет может выпасть.",
        answer=(
            "Во избежание выпадания пистолета из горловины бака переставьте автомобиль так, чтобы горловина топливного "
            "бака была со стороны топливораздаточной колонки."
        ),
    ),
    ScenarioId.WRONG_FUEL: SupportScenario(
        title="Выбран неправильный вид топлива",
        classifier_notes="Клиент ошибся с видом топлива или пытается оплатить не тот вид топлива.",
        answer="Вы выбрали неправильный вид топлива. Пожалуйста, поменяйте пистолет и повторите процедуру оплаты.",
    ),
    ScenarioId.WRONG_PAYMENT_METHOD: SupportScenario(
        title="Выбран неправильный способ оплаты",
        classifier_notes="Клиент выбрал не тот способ оплаты: наличные, банковская карта, топливная карта или дисконтная карта.",
        answer="Выбран неправильный способ оплаты. Пожалуйста, в меню терминала выберите соответствующий способ оплаты.",
    ),
    ScenarioId.DISCOUNT_CARD_ORIENTATION: SupportScenario(
        title="Как вставить дисконтную карту",
        classifier_notes="Клиент не понимает, какой стороной вставить дисконтную карту.",
        answer="Дисконтную карту нужно установить в картоприемник чипом вверх и вперед, до появления звукового сигнала.",
    ),
    ScenarioId.FUEL_CARD_ORIENTATION: SupportScenario(
        title="Как вставить топливную карту",
        classifier_notes="Клиент не понимает, какой стороной вставить топливную карту.",
        answer="Топливную карту нужно установить в картоприемник чипом вверх и вперед, до появления звукового сигнала.",
    ),
    ScenarioId.BANK_CARD_ORIENTATION: SupportScenario(
        title="Как вставить банковскую карту",
        classifier_notes="Клиент не понимает, какой стороной вставить банковскую карту.",
        answer="Банковскую карту нужно установить в картоприемник магнитной лентой вниз по правому краю карты.",
    ),
    ScenarioId.FUEL_NOT_FIT: SupportScenario(
        title="Оплаченный объем топлива не полностью вошел в бак",
        classifier_notes="Клиент оплатил больше, чем вошло в бак, бак полный, нужен чек возврата.",
        answer=(
            "Если заказанный объем топлива не полностью поместился в бак, потребуется оформление возврата. "
            "Сначала верните пистолет в исходное положение. Если нужна помощь с получением чека возврата, "
            "обратитесь к диспетчеру через кнопку связи в приложении."
        ),
    ),
    ScenarioId.FORBIDDEN_CONTAINER: SupportScenario(
        title="Заправка в запрещенную тару",
        classifier_notes="Клиент хочет залить топливо в пластиковую, открытую или стеклянную тару.",
        answer=(
            "В целях безопасности на АЗС запрещено заливать топливо в пластиковые, открытые и стеклянные емкости. "
            "При отказе выполнить это требование заправка топлива будет приостановлена."
        ),
    ),
    ScenarioId.FORGOT_CARD: SupportScenario(
        title="Карта забыта в картоприемнике",
        classifier_notes="Клиент забыл топливную или дисконтную карту в картоприемнике.",
        answer="Вы забыли карту в картоприемнике. Пожалуйста, заберите ее.",
    ),
    ScenarioId.FORGOT_NOZZLE: SupportScenario(
        title="Пистолет не возвращен в исходное положение",
        classifier_notes="Клиент забыл вернуть заправочный пистолет или собирается уехать с АЗС.",
        answer="Вы забыли вернуть пистолет в топливораздаточную колонку. Пожалуйста, верните его на место.",
    ),
    ScenarioId.TERMINAL_DISCOUNT_FAILURE: SupportScenario(
        title="Терминал не считывает дисконтную карту",
        classifier_notes="Технический сбой терминала: не считывается дисконтная карта.",
        answer=(
            "Произошел технический сбой оборудования, приносим извинения за неудобства. Если вас устроит, "
            "можно выполнить заправку за наличные деньги либо воспользоваться терминалом с другой стороны колонки."
        ),
    ),
    ScenarioId.TERMINAL_CARD_OR_RECEIPT_FAILURE: SupportScenario(
        title="Терминал не считывает банковскую карту или не печатает чек",
        classifier_notes="Технический сбой терминала: не печатается чек, не считывается банковская карта или другая похожая причина.",
        answer=(
            "Произошел технический сбой оборудования, приносим извинения за неудобства. Пожалуйста, воспользуйтесь "
            "терминалом с другой стороны колонки. Если проблема сохраняется, обратитесь к диспетчеру через кнопку связи "
            "в приложении."
        ),
    ),
    ScenarioId.GENERAL_REFUELING: SupportScenario(
        title="Общий порядок заправки на безоператорной АЗС",
        classifier_notes=(
            "Клиент задает общий вопрос: как заправиться, что делать сначала, как пользоваться безоператорной АЗС, "
            "какой общий порядок действий без уточнения способа оплаты."
        ),
        answer=(
            "Общий порядок такой: остановитесь у нужной колонки, заглушите двигатель, убедитесь, что выбран правильный "
            "вид топлива, вставьте пистолет в горловину бака и перейдите к терминалу оплаты. На терминале выберите способ "
            "оплаты, номер колонки, вид топлива и сумму или объем. После подтверждения оплаты выполните заправку, затем "
            "верните пистолет в колонку и заберите чек, если он нужен."
        ),
    ),
    ScenarioId.ENGINE_OFF_REQUIRED: SupportScenario(
        title="Нужно ли выключать двигатель во время заправки",
        classifier_notes=(
            "Клиент спрашивает, нужно ли глушить машину, можно ли заправляться с заведенным двигателем, "
            "можно ли оставлять двигатель включенным."
        ),
        answer=(
            "Да, перед началом заправки двигатель нужно заглушить. Это базовое требование безопасности на АЗС: "
            "не начинайте оплату и заправку, пока автомобиль не остановлен и двигатель не выключен."
        ),
    ),
    ScenarioId.STAY_NEAR_DISPENSER: SupportScenario(
        title="Можно ли находиться в машине во время заправки",
        classifier_notes=(
            "Клиент спрашивает, можно ли сидеть в машине, отходить от колонки или оставлять заправку без контроля."
        ),
        answer=(
            "Во время активной заправки не оставляйте процесс без контроля. Лучше находиться рядом с колонкой, следить "
            "за положением пистолета и не отходить от автомобиля. Двигатель при этом должен быть выключен. Если вы уже "
            "запустили заправку и возникла проблема, остановите процесс и свяжитесь с диспетчером через кнопку в приложении."
        ),
    ),
    ScenarioId.GENERAL_PAYMENT: SupportScenario(
        title="Общие способы оплаты на АЗС",
        classifier_notes=(
            "Клиент спрашивает, как оплатить заправку в целом, какие есть способы оплаты, где выбрать способ оплаты, "
            "без конкретной ошибки или сбоя."
        ),
        answer=(
            "На терминале выберите номер колонки, вид топлива и подходящий способ оплаты: банковская карта, топливная "
            "карта или наличные, если терминал поддерживает прием наличных. Затем укажите сумму или объем, подтвердите "
            "операцию и дождитесь сообщения об успешной оплате. После этого можно начинать заправку."
        ),
    ),
    ScenarioId.RECEIPT_GENERAL: SupportScenario(
        title="Получение чека после оплаты",
        classifier_notes="Клиент спрашивает, где взять чек, как получить чек, нужен ли чек или что делать, если чек не вышел.",
        answer=(
            "После успешной оплаты чек обычно печатается на терминале. Если терминал поддерживает электронный чек, "
            "следуйте подсказкам на экране. Если чек не напечатался или вам нужен документ для возврата, свяжитесь "
            "с диспетчером через кнопку в приложении."
        ),
    ),
    ScenarioId.NO_FUEL_FLOW: SupportScenario(
        title="Оплата прошла, но топливо не подается",
        classifier_notes=(
            "Клиент говорит, что оплатил, но колонка не включилась, топливо не идет, пистолет не подает топливо "
            "или заправка не началась."
        ),
        answer=(
            "Если оплата прошла, но топливо не подается, сначала проверьте, что выбран правильный номер колонки, "
            "пистолет плотно установлен в бак и рычаг пистолета нажат. Если заправка все равно не начинается, не пытайтесь "
            "разбирать оборудование и не переставляйте пистолет на другую колонку. Свяжитесь с диспетчером через кнопку "
            "в приложении."
        ),
    ),
    ScenarioId.SAFETY_PROHIBITIONS: SupportScenario(
        title="Базовые правила безопасности на АЗС",
        classifier_notes=(
            "Клиент спрашивает, можно ли курить, использовать открытый огонь, пользоваться телефоном рядом с колонкой, "
            "заправлять в канистру, нарушать правила безопасности или что запрещено на АЗС."
        ),
        answer=(
            "На территории АЗС нельзя курить, использовать открытый огонь и выполнять действия, которые могут создать "
            "искру или опасную ситуацию. Двигатель автомобиля должен быть выключен. Заправку нельзя оставлять без контроля. "
            "Топливо можно заливать только в бак автомобиля или в разрешенную для топлива тару, если это допускается правилами станции."
        ),
    ),
    ScenarioId.DISPATCHER_CONTACT_HELP: SupportScenario(
        title="Как связаться с диспетчером",
        classifier_notes=(
            "Клиент спрашивает, как позвать оператора, как обратиться к диспетчеру, где помощь человека, куда нажать при проблеме."
        ),
        answer=(
            "Чтобы связаться с диспетчером, нажмите кнопку «Связаться с диспетчером» в нижнем меню приложения. "
            "Система передаст обращение в диспетчерскую панель, а на экране появится номер для связи."
        ),
    ),
    ScenarioId.EMERGENCY_OR_UNSAFE: SupportScenario(
        title="Аварийная или небезопасная ситуация",
        classifier_notes="Клиент сообщает о пожаре, отключении электроэнергии, запахе топлива, разливе или противоправных действиях.",
        answer=(
            "Это похоже на нештатную или опасную ситуацию. Не продолжайте заправку и отойдите на безопасное расстояние. "
            "Пожалуйста, сразу свяжитесь с диспетчером через кнопку в приложении."
        ),
    ),
    ScenarioId.UNSUPPORTED: SupportScenario(
        title="Вопрос вне регламента",
        classifier_notes="Вопрос не относится к сценариям из стандарта или для ответа нет достаточных данных.",
        answer=(
            "Я не могу надежно ответить на этот вопрос по доступному регламенту. Пожалуйста, обратитесь к диспетчеру — "
            "для этого нажмите кнопку «Связаться с диспетчером» в приложении."
        ),
    ),
}


SCENARIO_CATALOG = "\n".join(
    f"- {scenario_id.value}: {scenario.title}. {scenario.classifier_notes}"
    for scenario_id, scenario in SCENARIOS.items()
)


class ScenarioClassification(BaseModel):
    scenario_id: ScenarioId
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(max_length=500)


class ScenarioReview(BaseModel):
    approved: bool
    corrected_scenario_id: ScenarioId
    reason: str = Field(max_length=500)


class AIState(TypedDict, total=False):
    station_code: str
    question: str
    classification: ScenarioClassification
    review: ScenarioReview
    answer: str


class LLMService:
    """Regulation-based LangGraph wrapper.

    The LLM classifies a client question and checks its own classification.
    The final answer is always selected from predefined standard-based replies.
    FAQ and dispatcher actions do not use this class.
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()
        self._llm = None
        self._graph = self._build_graph()

    def _get_llm(self) -> ChatDeepSeek:
        if not DEEPSEEK_API_KEY:
            raise HTTPException(status_code=500, detail="DeepSeek API key is not configured")
        if self._llm is None:
            self._llm = ChatDeepSeek(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                temperature=0,
                timeout=DEEPSEEK_TIMEOUT_SECONDS,
                max_retries=DEEPSEEK_MAX_RETRIES,
            )
        return self._llm

    @staticmethod
    def _extract_json(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            text = text.removesuffix("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text

    def _parse_model(self, model: type[BaseModel], content: str) -> BaseModel:
        return model.model_validate_json(self._extract_json(content))

    def _classify(self, question: str) -> ScenarioClassification:
        llm = self._get_llm()
        prompt = (
            "Ты классификатор обращений клиента безоператорной АЗС GasVision.\n"
            "Клиент стоит на станции у терминала или колонки и хочет быстро получить безопасную инструкцию.\n"
            "Твоя задача — выбрать один наиболее подходящий scenario_id из списка ниже. Ответ пользователю будет взят "
            "из заранее подготовленного регламента, поэтому не придумывай новые scenario_id и не добавляй свободный текст.\n"
            "Выбирай практичный ближайший сценарий, даже если клиент формулирует вопрос бытовыми словами.\n"
            "Например: «как заправиться», «что делать сначала», «как пользоваться колонкой» → general_refueling; "
            "«надо ли глушить машину», «можно с заведенным двигателем» → engine_off_required; "
            "«можно сидеть в машине», «можно отойти» → stay_near_dispenser; "
            "«как оплатить», «какие способы оплаты» → general_payment; "
            "«где чек», «как получить чек» → receipt_general; "
            "«оплатил, но бензин не идет» → no_fuel_flow; "
            "«можно курить», «что запрещено», «безопасность» → safety_prohibitions.\n"
            "Выбирай emergency_or_unsafe для запаха топлива, пожара, разлива, угрозы, агрессии, отключения электричества "
            "или другой опасной ситуации. Выбирай unsupported только если вопрос явно не относится к АЗС, оплате, "
            "заправке, безопасности, возврату, чеку, картам, терминалу или связи с диспетчером.\n\n"
            "Доступные scenario_id:\n"
            f"{SCENARIO_CATALOG}\n\n"
            "Верни строго JSON без markdown по схеме:\n"
            '{"scenario_id":"<one of ids>","confidence":0.0,"reason":"краткое объяснение"}'
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=question)])
        try:
            return self._parse_model(ScenarioClassification, response.content)  # type: ignore[return-value]
        except Exception:
            return ScenarioClassification(
                scenario_id=ScenarioId.UNSUPPORTED,
                confidence=0,
                reason="Не удалось надежно разобрать классификацию LLM.",
            )

    def _critique(self, question: str, classification: ScenarioClassification) -> ScenarioReview:
        llm = self._get_llm()
        prompt = (
            "Ты проверяешь классификацию обращения клиента безоператорной АЗС GasVision.\n"
            "Проверь, соответствует ли выбранный scenario_id смыслу вопроса и доступному регламенту.\n"
            "Если есть более точный сценарий из списка, укажи его. Не исправляй бытовые вопросы про заправку, оплату, "
            "чек, двигатель, нахождение у колонки, безопасность или связь с диспетчером на unsupported, если для них "
            "есть близкий сценарий. На unsupported исправляй только вопросы вне предметной области АЗС или случаи, "
            "где безопасный регламентный ответ действительно невозможен.\n"
            "Опасные ситуации — запах топлива, пожар, разлив, угроза, агрессия, отключение электричества — должны быть "
            "emergency_or_unsafe.\n\n"
            "Доступные scenario_id:\n"
            f"{SCENARIO_CATALOG}\n\n"
            "Верни строго JSON без markdown по схеме:\n"
            '{"approved":true,"corrected_scenario_id":"<one of ids>","reason":"краткое объяснение"}'
        )
        user_payload = (
            f"Вопрос клиента: {question}\n"
            f"Классификация: scenario_id={classification.scenario_id.value}, "
            f"confidence={classification.confidence}, reason={classification.reason}"
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_payload)])
        try:
            return self._parse_model(ScenarioReview, response.content)  # type: ignore[return-value]
        except Exception:
            return ScenarioReview(
                approved=classification.scenario_id == ScenarioId.UNSUPPORTED,
                corrected_scenario_id=classification.scenario_id,
                reason="Не удалось надежно разобрать проверку критика.",
            )

    def _build_graph(self):
        def classify_node(state: AIState):
            return {"classification": self._classify(state["question"])}

        def critique_node(state: AIState):
            return {"review": self._critique(state["question"], state["classification"])}

        def respond_node(state: AIState):
            classification = state["classification"]
            review = state["review"]
            scenario_id = review.corrected_scenario_id
            if classification.confidence < 0.45 and not review.approved:
                scenario_id = ScenarioId.UNSUPPORTED
            return {"answer": SCENARIOS[scenario_id].answer}

        graph = StateGraph(AIState)
        graph.add_node("classify", classify_node)
        graph.add_node("critique", critique_node)
        graph.add_node("respond", respond_node)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "critique")
        graph.add_edge("critique", "respond")
        graph.add_edge("respond", END)
        return graph.compile(checkpointer=self._checkpointer)

    def ask(self, *, station_code: str, session_id: str, question: str) -> str:
        """Classify a user question and return a standard-based answer."""

        result: dict[str, Any] = self._graph.invoke(
            {
                "station_code": station_code,
                "question": question,
            },
            config={"configurable": {"thread_id": session_id}},
        )
        return result["answer"]


llm_service = LLMService()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class BootstrapResponse(BaseModel):
    access_token: str
    station_code: str
    station_name: str
    session_id: str
    faq_items: list[dict]


class StationEntryResponse(BaseModel):
    access_token: str
    station_code: str
    station_name: str


class AskRequest(BaseModel):
    access_token: str
    session_id: str = Field(min_length=8)
    message: str = Field(min_length=1, max_length=3000)


class AskResponse(BaseModel):
    answer: str
    needs_feedback: bool = True


class FeedbackRequest(BaseModel):
    access_token: str
    session_id: str
    message: str
    answer: str
    helpful: bool


class FeedbackResponse(BaseModel):
    status: Literal["ok"]
    escalated: bool


class DispatcherRequest(BaseModel):
    access_token: str
    session_id: str


class DispatcherResponse(BaseModel):
    status: Literal["ok"]
    phone: str
    escalated: bool


class TranscribeResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def log_message(db: Session, *, session_id: str, station_code: str, source: str, role: str, content: str) -> None:
    db.add(
        ConversationMessage(
            session_id=session_id,
            station_code=station_code,
            source=source,
            role=role,
            content=content,
        )
    )
    db.commit()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_lock(774601)"))
        try:
            Base.metadata.create_all(bind=connection)
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(774601)"))
    yield


app = FastAPI(title="GasVision AI Support", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/public/station-entry", response_model=StationEntryResponse)
def station_entry(station_code: str):
    station = STATIONS.get(station_code)
    if station is None:
        raise HTTPException(status_code=404, detail="Unknown station")

    serializer = get_token_serializer()
    access_token = serializer.dumps({"station_code": station_code})
    return StationEntryResponse(
        access_token=access_token,
        station_code=station["station_code"],
        station_name=station["station_name"],
    )


@app.get("/api/public/bootstrap-station", response_model=BootstrapResponse)
def bootstrap_station(station_code: str):
    """Open the demo agent by station code and issue a short-lived access token."""

    station = STATIONS.get(station_code)
    if station is None:
        raise HTTPException(status_code=404, detail="Unknown station")

    serializer = get_token_serializer()
    access_token = serializer.dumps({"station_code": station_code})
    return BootstrapResponse(
        access_token=access_token,
        station_code=station["station_code"],
        station_name=station["station_name"],
        session_id=uuid.uuid4().hex,
        faq_items=FAQ_ITEMS,
    )


@app.get("/api/public/bootstrap", response_model=BootstrapResponse)
def bootstrap(access_token: str):
    """Validate QR access and return all data needed for initial screen render.

    No normal entry point exists without a valid QR token.
    """

    access = validate_qr_token(access_token)
    session_id = uuid.uuid4().hex
    return BootstrapResponse(
        access_token=access_token,
        station_code=access.station_code,
        station_name=access.station_name,
        session_id=session_id,
        faq_items=FAQ_ITEMS,
    )


@app.post("/api/public/ask", response_model=AskResponse)
def ask_ai(payload: AskRequest, db: Annotated[Session, Depends(get_db)]):
    access = validate_qr_token(payload.access_token)

    # Persist user question for audit / demo traceability.
    log_message(
        db,
        session_id=payload.session_id,
        station_code=access.station_code,
        source="ai",
        role="user",
        content=payload.message,
    )

    answer = llm_service.ask(
        station_code=access.station_code,
        session_id=payload.session_id,
        question=payload.message,
    )

    # Persist assistant answer as well.
    log_message(
        db,
        session_id=payload.session_id,
        station_code=access.station_code,
        source="ai",
        role="assistant",
        content=answer,
    )

    return AskResponse(answer=answer)


@app.post("/api/public/feedback", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest, db: Annotated[Session, Depends(get_db)]):
    access = validate_qr_token(payload.access_token)

    # Store explicit feedback so later it can be analysed.
    feedback_text = "Помог ответ" if payload.helpful else "Ответ не помог"
    log_message(
        db,
        session_id=payload.session_id,
        station_code=access.station_code,
        source="feedback",
        role="user",
        content=feedback_text,
    )

    escalated = False
    if not payload.helpful:
        escalated = await event_client.create_escalation(
            title="AI escalation: клиенту не помог ответ AI",
            station_code=access.station_code,
        )

    return FeedbackResponse(status="ok", escalated=escalated)


@app.post("/api/public/contact-dispatcher", response_model=DispatcherResponse)
async def contact_dispatcher(payload: DispatcherRequest, db: Annotated[Session, Depends(get_db)]):
    access = validate_qr_token(payload.access_token)

    # Dispatcher request is intentionally non-LLM flow.
    log_message(
        db,
        session_id=payload.session_id,
        station_code=access.station_code,
        source="dispatcher",
        role="user",
        content="Клиент запросил связь с диспетчером",
    )

    escalated = await event_client.create_escalation(
        title="AI escalation: клиент запросил связь с диспетчером",
        station_code=access.station_code,
    )

    return DispatcherResponse(status="ok", phone=DISPATCHER_PHONE, escalated=escalated)


@app.post("/api/public/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    access_token: Annotated[str, Form()],
    session_id: Annotated[str, Form(min_length=8)],
    audio: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
):
    access = validate_qr_token(access_token)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(audio_bytes) > VOICE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large")

    text = await speechkit_client.transcribe(audio_bytes)
    if not text:
        raise HTTPException(status_code=422, detail="Speech was not recognized")

    log_message(
        db,
        session_id=session_id,
        station_code=access.station_code,
        source="speech_to_text",
        role="user",
        content=text,
    )
    return TranscribeResponse(text=text)
