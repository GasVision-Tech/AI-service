import React, { useEffect, useRef, useState } from 'react'
import microphoneIcon from './assets/microphone.png'
import sendIcon from './assets/send.png'

const API_BASE = '/api'

async function apiGet(path, params = {}) {
  const url = new URL(`${window.location.origin}${API_BASE}${path}`)
  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value))
  const response = await fetch(url.toString())
  if (!response.ok) throw new Error('Request failed')
  return response.json()
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error('Request failed')
  return response.json()
}

async function apiPostForm(path, formData) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) throw new Error('Request failed')
  return response.json()
}

function MessageBubble({ item, onFeedback, feedbackBusy }) {
  return (
    <div className={`message-row ${item.role}`}>
      <div className={`message ${item.role} ${item.kind ? `message-${item.kind}` : ''}`}>
        <div>{item.text}</div>

        {item.kind === 'feedback' && (
          <div className="inline-feedback">
            <button
              className="inline-feedback-btn"
              onClick={() => onFeedback(true)}
              disabled={feedbackBusy}
            >
              Да
            </button>
            <button
              className="inline-feedback-btn negative"
              onClick={() => onFeedback(false)}
              disabled={feedbackBusy}
            >
              Нет
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="message assistant typing-message">
        <div className="typing-dots">
          <span />
          <span />
          <span />
        </div>
      </div>
    </div>
  )
}

function createEphemeralSessionId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function createMessageId() {
  return globalThis.crypto?.randomUUID?.() || `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function getSupportedAudioMimeType() {
  const candidates = [
    'audio/ogg;codecs=opus',
    'audio/webm;codecs=opus',
    'audio/ogg',
    'audio/webm',
  ]
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || ''
}

const STATION_ENTRIES = [
  {
    stationCode: 'AZS-001',
    stationName: 'АЗС #101',
    location: 'Казань, пр-т Победы',
    path: '/tatneft-azs-001',
  },
]

export default function App() {
  const [boot, setBoot] = useState({
    loading: true,
    allowed: false,
    stationCode: '',
    stationName: '',
    sessionId: '',
    faqItems: [],
  })

  const [mode, setMode] = useState('faq')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [awaitingFeedback, setAwaitingFeedback] = useState(false)
  const [lastAiExchange, setLastAiExchange] = useState(null)
  const [showFaqPanel, setShowFaqPanel] = useState(true)
  const [entryMode, setEntryMode] = useState('agent')
  const [accessToken, setAccessToken] = useState('')
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)

  const bottomAnchorRef = useRef(null)
  const chatBoxRef = useRef(null)
  const inputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const audioChunksRef = useRef([])
  const recordingTimerRef = useRef(null)

  function resetBoot(next = {}) {
    setBoot({
      loading: false,
      allowed: false,
      stationCode: '',
      stationName: '',
      sessionId: '',
      faqItems: [],
      ...next,
    })
  }

  function applyBootstrap(data) {
    setAccessToken(data.access_token || '')
    setEntryMode('agent')
    setBoot({
      loading: false,
      allowed: true,
      stationCode: data.station_code,
      stationName: data.station_name,
      sessionId: data.session_id || createEphemeralSessionId(),
      faqItems: data.faq_items || [],
    })
    setMessages([
      {
        id: createMessageId(),
        role: 'assistant',
        text: `Здравствуйте! Вы подключены к станции ${data.station_code}. Чем могу помочь?`,
      },
    ])
  }

  async function loadStation(stationCode) {
    setEntryMode('agent')
    setBoot((state) => ({ ...state, loading: true }))
    try {
      const data = await apiGet('/public/bootstrap-station', { station_code: stationCode })
      applyBootstrap(data)
    } catch {
      setEntryMode('selector')
      resetBoot()
    }
  }

  async function loadAccessToken(token) {
    setEntryMode('agent')
    setBoot((state) => ({ ...state, loading: true }))
    try {
      const data = await apiGet('/public/bootstrap', { access_token: token })
      applyBootstrap({ ...data, access_token: data.access_token || token })
    } catch {
      setEntryMode('selector')
      resetBoot()
    }
  }

  async function startStation(stationCode, updateUrl = true) {
    if (updateUrl) {
      const station = STATION_ENTRIES.find((item) => item.stationCode === stationCode)
      const nextPath = station?.path || '/'
      window.history.pushState({}, '', nextPath)
    }

    await loadStation(stationCode)
  }

  useEffect(() => {
    const urlToken = new URLSearchParams(window.location.search).get('access_token') || ''
    const path = window.location.pathname.replace(/\/$/, '') || '/'
    const station = STATION_ENTRIES.find((item) => item.path === path)

    if (urlToken) {
      loadAccessToken(urlToken)
      return
    }

    if (station) {
      loadStation(station.stationCode)
      return
    }

    setEntryMode('selector')
    resetBoot()
  }, [])

  useEffect(() => {
    async function bootstrap() {
      if (!accessToken || boot.allowed) return

      setEntryMode('agent')
      setBoot((state) => ({ ...state, loading: true }))
      try {
        const data = await apiGet('/public/bootstrap', { access_token: accessToken })
        setBoot({
          loading: false,
          allowed: true,
          stationCode: data.station_code,
          stationName: data.station_name,
          sessionId: data.session_id || createEphemeralSessionId(),
          faqItems: data.faq_items || [],
        })
        setMessages([
          {
            id: createMessageId(),
            role: 'assistant',
            text: `Здравствуйте! Вы подключены к станции ${data.station_code}. Чем могу помочь?`,
          },
        ])
      } catch {
        setEntryMode('selector')
        resetBoot()
      }
    }

    bootstrap()
  }, [accessToken, boot.allowed])

  useEffect(() => {
    function handlePopState() {
      const path = window.location.pathname.replace(/\/$/, '') || '/'
      if (path === '/') {
        setAccessToken('')
        setEntryMode('selector')
        resetBoot()
      }
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearTimeout(recordingTimerRef.current)
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  useEffect(() => {
    const chatBox = chatBoxRef.current
    if (!chatBox) return

    requestAnimationFrame(() => {
      chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' })
    })
  }, [messages, busy, showFaqPanel, mode])

  useEffect(() => {
    if (mode === 'ask' && inputRef.current) {
      const timer = setTimeout(() => inputRef.current?.focus(), 180)
      return () => clearTimeout(timer)
    }
  }, [mode])

  useEffect(() => {
    const field = inputRef.current
    if (!field) return

    field.style.height = 'auto'
    field.style.height = `${Math.min(field.scrollHeight, 172)}px`
  }, [input])

  function appendMessage(message) {
    setMessages((prev) => [...prev, { id: createMessageId(), ...message }])
  }

  function removeFeedbackMessage() {
    setMessages((prev) => prev.filter((item) => item.kind !== 'feedback'))
  }

  async function handleFaqClick(item) {
    appendMessage({ role: 'user', text: item.question })
    appendMessage({ role: 'assistant', text: item.answer })
    setAwaitingFeedback(false)
    setLastAiExchange(null)
    setMode('faq')
    setShowFaqPanel(false)
  }

  async function handleAsk() {
    const question = input.trim()
    if (!question || busy || transcribing) return

    setBusy(true)
    setAwaitingFeedback(false)
    removeFeedbackMessage()

    appendMessage({ role: 'user', text: question })
    setInput('')
    setShowFaqPanel(false)
    setMode('ask')

    try {
      const data = await apiPost('/public/ask', {
        access_token: accessToken,
        session_id: boot.sessionId,
        message: question,
      })

      appendMessage({ role: 'assistant', text: data.answer })
      appendMessage({
        role: 'assistant',
        text: 'Помог ли вам этот ответ?',
        kind: 'feedback',
      })

      setLastAiExchange({ message: question, answer: data.answer })
      setAwaitingFeedback(true)
    } catch {
      appendMessage({
        role: 'assistant',
        text: 'Я не могу надежно ответить на этот вопрос по доступному регламенту. Пожалуйста, обратитесь к диспетчеру — для этого нажмите кнопку «Связаться с диспетчером» в приложении.',
      })
      setLastAiExchange(null)
      setAwaitingFeedback(false)
    } finally {
      setBusy(false)
    }
  }

  async function handleFeedback(helpful) {
    if (!lastAiExchange || busy) return

    setBusy(true)

    try {
      const data = await apiPost('/public/feedback', {
        access_token: accessToken,
        session_id: boot.sessionId,
        message: lastAiExchange.message,
        answer: lastAiExchange.answer,
        helpful,
      })

      removeFeedbackMessage()

      if (helpful) {
        appendMessage({
          role: 'assistant',
          text: 'Спасибо за обратную связь.',
        })
      } else {
        appendMessage({
          role: 'assistant',
          text: data.escalated
            ? 'Запрос эскалирован диспетчеру. При необходимости используйте кнопку связи с диспетчером.'
            : 'Не удалось передать эскалацию в систему событий, но вы можете связаться с диспетчером вручную.',
        })
      }
    } catch {
      removeFeedbackMessage()
      appendMessage({
        role: 'assistant',
        text: 'Не удалось отправить обратную связь.',
      })
    } finally {
      setBusy(false)
      setLastAiExchange(null)
      setAwaitingFeedback(false)
    }
  }

  async function handleDispatcher() {
    if (busy || transcribing) return

    setBusy(true)
    setShowFaqPanel(false)
    removeFeedbackMessage()
    setAwaitingFeedback(false)

    try {
      const data = await apiPost('/public/contact-dispatcher', {
        access_token: accessToken,
        session_id: boot.sessionId,
      })

      appendMessage({ role: 'user', text: 'Связаться с диспетчером' })
      appendMessage({
        role: 'assistant',
        text: `Номер диспетчера: ${data.phone}`,
      })

      setLastAiExchange(null)
      setMode('dispatcher')
    } catch {
      appendMessage({
        role: 'assistant',
        text: 'Не удалось получить номер диспетчера.',
      })
    } finally {
      setBusy(false)
    }
  }

  function openFaqPanel() {
    setMode('faq')
    setShowFaqPanel((prev) => !prev)
  }

  function openAskMode() {
    setMode('ask')
    setShowFaqPanel(false)
  }

  function stopRecording() {
    if (recordingTimerRef.current) {
      clearTimeout(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
  }

  async function transcribeAudio(blob) {
    setTranscribing(true)
    try {
      const formData = new FormData()
      formData.append('access_token', accessToken)
      formData.append('session_id', boot.sessionId)
      formData.append('audio', blob, 'question-audio.webm')
      const data = await apiPostForm('/public/transcribe', formData)
      setInput(data.text || '')
      setMode('ask')
      setShowFaqPanel(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    } catch {
      appendMessage({
        role: 'assistant',
        text: 'Не удалось распознать голос. Попробуйте еще раз или введите вопрос текстом.',
      })
    } finally {
      setTranscribing(false)
    }
  }

  async function toggleRecording() {
    if (recording) {
      stopRecording()
      return
    }

    if (busy || transcribing || awaitingFeedback) return

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      appendMessage({
        role: 'assistant',
        text: 'Запись голоса доступна только в браузерах с поддержкой микрофона и обычно требует HTTPS. Пока можно ввести вопрос текстом.',
      })
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = getSupportedAudioMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)

      audioChunksRef.current = []
      mediaRecorderRef.current = recorder
      mediaStreamRef.current = stream

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data)
      }

      recorder.onstop = () => {
        setRecording(false)
        stream.getTracks().forEach((track) => track.stop())
        mediaStreamRef.current = null

        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })
        audioChunksRef.current = []
        if (audioBlob.size > 0) transcribeAudio(audioBlob)
      }

      recorder.start()
      setRecording(true)
      setMode('ask')
      setShowFaqPanel(false)
      recordingTimerRef.current = setTimeout(stopRecording, 30000)
    } catch {
      appendMessage({
        role: 'assistant',
        text: 'Не получилось получить доступ к микрофону. Проверьте разрешения браузера или введите вопрос текстом.',
      })
    }
  }

  if (boot.loading) {
    return <div className="center-screen">Загрузка...</div>
  }

  if (entryMode === 'selector') {
    return (
      <div className="center-screen station-entry-screen">
        <div className="station-entry-card">
          <div className="station-entry-kicker">GasVision поддержка</div>
          <h1>Выберите АЗС</h1>
          <div className="station-entry-list">
            {STATION_ENTRIES.map((station) => (
              <button
                key={station.stationCode}
                className="station-entry-button"
                onClick={() => startStation(station.stationCode)}
              >
                <span className="station-entry-name">{station.stationName}</span>
                <span className="station-entry-location">{station.location}</span>
                <span className="station-entry-code">Код станции: {station.stationCode}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (entryMode === 'agent' && !boot.allowed) {
    return <div className="center-screen">Загрузка...</div>
  }

  if (false && entryMode === 'denied') {
    return (
      <div className="center-screen denied">
        <div className="denied-card">
          <h1>GasVision поддержка</h1>
          <p>Не удалось открыть станцию. Вернитесь к выбору АЗС и попробуйте снова.</p>
        </div>
      </div>
    )
  }

  const composerVisible = mode === 'ask'

  return (
    <div className="page-shell">
      <div className="app-shell">
        <header className="header">
          <div className="header-title">GasVision поддержка</div>
          <div className="header-station">
            {boot.stationName} · {boot.stationCode}
          </div>
        </header>

        <main
          className={`content ${composerVisible ? 'with-composer' : ''} ${
            showFaqPanel ? 'with-faq' : ''
          }`}
        >
          <section className="chat-section">
            <div className="chat-box" ref={chatBoxRef}>
              {messages.map((item) => (
                <MessageBubble
                  key={item.id}
                  item={item}
                  onFeedback={handleFeedback}
                  feedbackBusy={busy}
                />
              ))}

              {busy && mode === 'ask' && <TypingIndicator />}
              <div ref={bottomAnchorRef} />
            </div>
          </section>
        </main>

        <div className={`composer-dock ${composerVisible ? 'visible' : ''}`}>
          <div className="composer">
            <textarea
              ref={inputRef}
              className="composer-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Например: как заправиться по банковской карте?"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleAsk()
                }
              }}
              disabled={busy || transcribing || recording}
            />
            <button
              className={`composer-mic ${recording ? 'recording' : ''}`}
              onClick={toggleRecording}
              disabled={busy || transcribing || awaitingFeedback}
              title={recording ? 'Остановить запись' : 'Задать вопрос голосом'}
              aria-label={recording ? 'Остановить запись' : 'Задать вопрос голосом'}
            >
              {transcribing ? (
                <span className="mic-loading" aria-hidden="true" />
              ) : (
                <img className="mic-icon" src={microphoneIcon} alt="" aria-hidden="true" />
              )}
            </button>
            <button
              className="composer-send"
              onClick={handleAsk}
              title={busy ? 'Отправка...' : transcribing ? 'Распознаю...' : 'Отправить'}
              aria-label={busy ? 'Отправка...' : transcribing ? 'Распознаю...' : 'Отправить'}
              disabled={busy || transcribing || recording || !input.trim() || awaitingFeedback}
            >
              {busy || transcribing ? (
                <span className="send-loading" aria-hidden="true" />
              ) : (
                <img className="send-icon" src={sendIcon} alt="" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>

        <div className={`faq-panel ${showFaqPanel ? 'open' : ''}`}>
          <div className="faq-list">
            <div className="faq-panel-title">Частые вопросы</div>

            {boot.faqItems.map((item) => (
              <button
                key={item.id}
                className="faq-button"
                onClick={() => handleFaqClick(item)}
              >
                {item.question}
              </button>
            ))}
          </div>
        </div>

        <nav className="bottom-nav">
          <button
            className={`nav-btn nav-btn-faq ${mode === 'faq' ? 'active' : ''}`}
            onClick={openFaqPanel}
          >
            FAQ
          </button>

          <button
            className={`nav-btn nav-btn-ai ${mode === 'ask' ? 'active' : ''}`}
            onClick={openAskMode}
          >
            Спросить AI
          </button>

          <button
            className={`nav-btn nav-btn-dispatcher ${mode === 'dispatcher' ? 'active' : ''}`}
            onClick={handleDispatcher}
          >
            Связь с диспетчером
          </button>
        </nav>
      </div>
    </div>
  )
}
