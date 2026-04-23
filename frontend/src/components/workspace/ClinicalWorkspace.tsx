import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Shield } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'

import { api } from '../../services/api'
import {
  clamp,
  formatDate,
  getProfileStorageKey,
  getSeverityTone,
  readStorageItem,
  removeStorageItem,
  writeStorageItem,
} from '../../lib/clinical-utils'
import { useAnalysisWorkflow } from '../../hooks/useAnalysisWorkflow'
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts'
import { useProfileSwitchNotice } from '../../hooks/useProfileSwitchNotice'
import { useStatusPoller } from '../../hooks/useStatusPoller'
import { useToast } from '../../hooks/useToast'
import { useWorkspacePrefs } from '../../hooks/useWorkspacePrefs'
import type {
  ComparePayload,
  ConsensusLesion,
  PrivacyConfig,
  ProfileSummary,
  RegionStats,
  SessionDetail,
  SessionSummary,
} from '../../types/api'
import { MetricCard } from './MetricCard'
import { SessionLoadingSkeleton } from './Skeleton'
import type { ViewerState } from './types'

import { CaseComparePanel } from './CaseComparePanel'
import { ClinicalGradePanel } from './ClinicalGradePanel'
import { CompareSection } from './CompareSection'
import { DetectionCanvas } from './DetectionCanvas'
import { LesionTable } from './LesionTable'
import { OnboardingModal } from './OnboardingModal'
import { PrivacyExportPanel } from './PrivacyExportPanel'
import { SessionArchiveRail } from './SessionArchiveRail'
import { SessionNotesPanel } from './SessionNotesPanel'
import { SystemStateRail } from './SystemStateRail'
import { UploadPanel } from './UploadPanel'

type CompareRegionDelta = NonNullable<ComparePayload>['regions'][string]
const BASELINE_SESSION_KEY = 'clearskin-baseline-session'
const ONBOARDING_SEEN_KEY = 'clearskin-ui-prefs-onboarding-seen'
const DEFAULT_PROFILE_ID = 'default-profile'

export function ClinicalWorkspace() {
  const [caseState, dispatch] = useAnalysisWorkflow()
  const {
    isAnalyzing, selectedFile, previewUrl, sessionId,
    active, compare, previousSession, noteDraft,
    activeLesionKey, profileGuard, error,
    isExporting, isSavingNotes, isLoadingSession, isPurging,
  } = caseState

  const [history, setHistory] = useState<SessionSummary[]>([])
  const [historyCursor, setHistoryCursor] = useState<string | null>(null)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [profiles, setProfiles] = useState<ProfileSummary[]>([])
  const [activeProfileId, setActiveProfileId] = useState<string>(DEFAULT_PROFILE_ID)
  const [baselineSession, setBaselineSession] = useState<SessionSummary | null>(null)
  const [privacy, setPrivacy] = useState<PrivacyConfig | null>(null)
  const [privacyMode, setPrivacyMode] = useState(false)
  const [retentionHours, setRetentionHours] = useState(72)
  const [compareFullscreen, setCompareFullscreen] = useState(false)
  const [mainViewer, setMainViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [compareViewer, setCompareViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [lesionRegionFilter, setLesionRegionFilter] = useState<string>('all')
  const [lesionConfidenceFilter, setLesionConfidenceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [lesionTypeFilter, setLesionTypeFilter] = useState<string>('all')
  const [showOnboarding, setShowOnboarding] = useState(false)
  const workspaceRef = useRef<HTMLDivElement | null>(null)
  const resultsRef = useRef<HTMLDivElement | null>(null)

  const { toast } = useToast()

  const {
    leftRailWidth, setLeftRailWidth,
    rightRailWidth, setRightRailWidth,
    compareViewerMode, setCompareViewerMode,
    showDetectionOverlay, setShowDetectionOverlay,
    showCompareOverlay, setShowCompareOverlay,
    exportPreset, setExportPreset,
  } = useWorkspacePrefs(activeProfileId)

  const [status, setStatus] = useStatusPoller(sessionId, isAnalyzing)
  const profileSwitchNotice = useProfileSwitchNotice(activeProfileId)

  // ---------- Effects ----------

  useEffect(() => {
    setShowOnboarding(readStorageItem(ONBOARDING_SEEN_KEY) !== 'true')

    let activeRequest = true

    void api.getPrivacy()
      .then((privacyConfig) => {
        if (!activeRequest) return
        setPrivacy(privacyConfig)
        setRetentionHours(privacyConfig.default_retention_hours)
      })
      .catch((err) => {
        if (activeRequest) {
          dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to initialize workspace' })
        }
      })

    return () => { activeRequest = false }
  }, [])

  useEffect(() => {
    setBaselineSession(null)

    let activeRequest = true
    const baselineStorageKey = getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId)

    void Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
      .then(([historyPage, profileItems]) => {
        if (!activeRequest) return
        setHistory(historyPage.items)
        setHistoryCursor(historyPage.next_cursor ?? null)
        setProfiles(profileItems)

        const storedBaseline = readStorageItem(baselineStorageKey)
        if (!storedBaseline) return

        const match = historyPage.items.find((item) => item.session_id === storedBaseline)
        if (match) {
          setBaselineSession(match)
          return
        }
        removeStorageItem(baselineStorageKey)
      })
      .catch((err) => {
        if (activeRequest) {
          dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to load profile workspace' })
        }
      })

    return () => { activeRequest = false }
  }, [activeProfileId])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const onFullScreenChange = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', onFullScreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullScreenChange)
  }, [])

  // ---------- Derived state ----------

  const regionDeltaCards = useMemo<[string, CompareRegionDelta][]>(() => {
    if (!compare?.regions) return []
    return (Object.entries(compare.regions) as [string, CompareRegionDelta][])
      .sort((a, b) => Math.abs(b[1].count_delta) - Math.abs(a[1].count_delta))
      .slice(0, 6)
  }, [compare])

  const isPinnedBaselineCompare = Boolean(compare && baselineSession?.session_id === compare.previous_session_id)
  const isViewingPinnedBaseline = Boolean(active?.session_id && baselineSession?.session_id === active.session_id)

  const consensusLesions = active?.results?.consensus_summary?.lesions ?? []
  const clinicalAnalysis = active?.results?.clinical_analysis
  const displaySeverity = active?.severity ?? clinicalAnalysis?.clinical_severity ?? 'Unknown'
  const displayGags = active?.gags_score ?? clinicalAnalysis?.gags_total_score ?? 0
  const displayLesions = active?.lesion_count ?? clinicalAnalysis?.total_lesions ?? 0
  const displaySymmetry = active?.symmetry_delta ?? clinicalAnalysis?.symmetry_delta ?? 0
  const severityTone = getSeverityTone(displaySeverity)

  const regionRows = useMemo(
    () => (Object.entries(clinicalAnalysis?.regions ?? {}) as [string, RegionStats][]).sort(
      (a, b) => (b[1].gags_score ?? 0) - (a[1].gags_score ?? 0),
    ),
    [clinicalAnalysis],
  )

  const activeImage = showDetectionOverlay ? active?.diagnostic_image ?? active?.original_image ?? '' : active?.original_image ?? active?.diagnostic_image ?? ''
  const previousImage = showCompareOverlay
    ? previousSession?.diagnostic_image ?? previousSession?.original_image ?? ''
    : previousSession?.original_image ?? previousSession?.diagnostic_image ?? ''

  const lesionOverlayItems = useMemo(
    () => consensusLesions.map((lesion, index) => ({
      ...lesion,
      key: `${lesion.region}-${index}`,
    })),
    [consensusLesions],
  )

  const filteredLesionOverlayItems = useMemo(
    () => lesionOverlayItems.filter((lesion) => {
      const regionPass = lesionRegionFilter === 'all' || lesion.region === lesionRegionFilter
      const confidencePass =
        lesionConfidenceFilter === 'all'
        || (lesionConfidenceFilter === 'high' && lesion.confidence >= 0.7)
        || (lesionConfidenceFilter === 'medium' && lesion.confidence >= 0.4 && lesion.confidence < 0.7)
        || (lesionConfidenceFilter === 'low' && lesion.confidence < 0.4)
      const typePass = lesionTypeFilter === 'all' || (lesion.class_name ?? 'acne') === lesionTypeFilter
      return regionPass && confidencePass && typePass
    }),
    [lesionOverlayItems, lesionRegionFilter, lesionConfidenceFilter, lesionTypeFilter],
  )

  const lesionRegions = useMemo(
    () => Array.from(new Set(lesionOverlayItems.map((lesion) => lesion.region))).sort(),
    [lesionOverlayItems],
  )
  const lesionTypes = useMemo(
    () => Array.from(new Set(lesionOverlayItems.map((lesion) => lesion.class_name ?? 'acne'))).sort(),
    [lesionOverlayItems],
  )
  const activeProfileSummary = profiles.find((profile) => profile.profile_id === activeProfileId) ?? null
  const activeSessionProfileId = active?.profile_id
    ?? history.find((item) => item.session_id === active?.session_id)?.profile_id
    ?? DEFAULT_PROFILE_ID

  // ---------- Handlers ----------

  const refreshHistory = useCallback(async () => {
    const [historyPage, profileItems] = await Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
    setHistory(historyPage.items)
    setHistoryCursor(historyPage.next_cursor ?? null)
    setProfiles(profileItems)
  }, [activeProfileId])

  const loadMoreHistory = useCallback(async () => {
    if (!historyCursor || isLoadingMore) return
    setIsLoadingMore(true)
    try {
      const page = await api.getHistory(30, activeProfileId, historyCursor)
      setHistory(prev => [...prev, ...page.items])
      setHistoryCursor(page.next_cursor ?? null)
    } catch {
      // Silently fail — user can retry by scrolling again
    } finally {
      setIsLoadingMore(false)
    }
  }, [historyCursor, isLoadingMore, activeProfileId])

  const loadCompareContext = async (
    currentSessionId: string,
    options?: {
      previousSessionId?: string | null
      fallbackCompare?: ComparePayload
    },
  ) => {
    const requestedPreviousSessionId = options?.previousSessionId
    let nextCompare = options?.fallbackCompare ?? null

    if (requestedPreviousSessionId) {
      nextCompare = requestedPreviousSessionId === currentSessionId
        ? null
        : await api.getCompare(currentSessionId, requestedPreviousSessionId)
    } else if (!nextCompare) {
      nextCompare = await api.getCompare(currentSessionId)
    }

    let nextPrevious: SessionDetail | null = null
    if (nextCompare?.previous_session_id) {
      nextPrevious = await api.getSession(nextCompare.previous_session_id)
    }

    dispatch({ type: 'SET_COMPARE', compare: nextCompare, previousSession: nextPrevious })
    return nextCompare
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    dispatch({
      type: 'SELECT_FILE',
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
    })
  }

  const handleStart = async () => {
    if (!selectedFile) return
    try {
      dispatch({ type: 'START_ANALYSIS' })
      setStatus({
        stage: 'warming_up',
        detail: 'Waking backend — this may take a moment on first use...',
        progress: 1,
      })

      try {
        await api.wakeBackend()
      } catch {
        // Health check failed — continue anyway
      }

      setStatus({
        stage: 'warming_up',
        detail: 'Booting segmentation + cloud engines for first analysis',
        progress: 3,
      })
      const start = await api.startAnalysis({
        profile_id: activeProfileId,
        privacy_mode: privacyMode,
        retention_hours: retentionHours,
      })
      const resolvedProfileId = start.profile_id ?? activeProfileId
      dispatch({ type: 'SET_SESSION', sessionId: start.session_id })
      if (resolvedProfileId !== activeProfileId) {
        setActiveProfileId(resolvedProfileId)
      }
      setStatus(start.status)

      const form = new FormData()
      form.append('file', selectedFile)
      form.append('session_id', start.session_id)
      form.append('profile_id', resolvedProfileId)
      form.append('privacy_mode', String(privacyMode))
      form.append('retention_hours', String(retentionHours))

      const result = await api.analyze(form)
      dispatch({ type: 'ANALYSIS_COMPLETE', active: result })
      await loadCompareContext(result.session_id, {
        previousSessionId: baselineSession?.session_id,
        fallbackCompare: result.compare,
      })
      setStatus(result.status)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
      await refreshHistory()

      toast('Analysis complete — results are ready for review.', { type: 'success' })

      // Scroll to results after a short delay to allow render
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (err) {
      const message = err instanceof Error
        ? err.message
        : 'Analysis failed — please try again'
      dispatch({ type: 'ANALYSIS_FAILED', error: message })
      toast(message, { type: 'error' })
    } finally {
      dispatch({ type: 'ANALYSIS_FINISHED' })
    }
  }

  const handleSelectHistory = async (item: SessionSummary) => {
    try {
      dispatch({ type: 'LOAD_SESSION_START' })
      const session = await api.getSession(item.session_id)
      const sessionProfileId = item.profile_id ?? DEFAULT_PROFILE_ID
      if (sessionProfileId !== activeProfileId) {
        setActiveProfileId(sessionProfileId)
      }
      const comparePayload = await loadCompareContext(item.session_id, {
        previousSessionId: baselineSession?.session_id,
      })
      dispatch({
        type: 'LOAD_SESSION_COMPLETE',
        sessionId: item.session_id,
        active: {
          session_id: session.session_id,
          status: session.status ?? { stage: 'completed', detail: 'Loaded from archive', progress: 100 },
          severity: session.severity,
          gags_score: session.gags_score,
          lesion_count: session.lesion_count,
          symmetry_delta: session.symmetry_delta,
          results: session.results ?? {},
          compare: comparePayload,
          diagnostic_image: session.diagnostic_image ?? null,
          original_image: session.original_image ?? null,
        },
        noteDraft: session.note ?? '',
      })
      setStatus(session.status ?? null)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
    } catch (err) {
      dispatch({ type: 'LOAD_SESSION_FAILED', error: err instanceof Error ? err.message : 'Failed to load session' })
    }
  }

  const handlePurge = async (item: SessionSummary) => {
    if (!window.confirm('Permanently delete this session? This cannot be undone.')) return
    try {
      dispatch({ type: 'PURGE_START' })
      await api.purgeSession(item.session_id)
      if (active?.session_id === item.session_id) {
        dispatch({ type: 'PURGE_ACTIVE_CLEARED' })
        setStatus(null)
      }
      await refreshHistory()
      dispatch({ type: 'PURGE_COMPLETE' })
      toast('Session purged from archive.', { type: 'success' })
    } catch (err) {
      dispatch({ type: 'PURGE_FAILED', error: err instanceof Error ? err.message : 'Failed to purge session' })
      toast('Failed to purge session.', { type: 'error' })
    }
  }

  const resetCaseWorkspace = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    dispatch({ type: 'RESET_CASE' })
    setStatus(null)
  }

  const handleExport = async () => {
    if (!active?.session_id) return
    try {
      dispatch({ type: 'EXPORT_START' })
      const bundle = await api.exportBundle(active.session_id, exportPreset, compare?.previous_session_id)
      if (bundle.pdf_data_uri) {
        // Async conversion — avoids blocking main thread with synchronous atob + byte loop
        const response = await fetch(bundle.pdf_data_uri)
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `acne_report_${active.session_id.slice(0, 8)}_${exportPreset}.pdf`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
      dispatch({ type: 'EXPORT_COMPLETE' })
      toast(`${exportPreset.charAt(0).toUpperCase() + exportPreset.slice(1)} report downloaded.`, { type: 'success' })
    } catch (err) {
      dispatch({ type: 'EXPORT_FAILED', error: err instanceof Error ? err.message : 'Export failed' })
      toast('Export failed — please try again.', { type: 'error' })
    }
  }

  const pinBaseline = async (item: SessionSummary) => {
    setBaselineSession(item)
    writeStorageItem(getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId), item.session_id)

    if (active?.session_id) {
      try {
        await loadCompareContext(active.session_id, { previousSessionId: item.session_id })
      } catch (err) {
        dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to load baseline comparison' })
      }
    }
    toast(`Baseline pinned to session #${item.session_id.slice(0, 8)}.`, { type: 'info' })
  }

  const clearBaseline = async () => {
    setBaselineSession(null)
    removeStorageItem(getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId))

    if (active?.session_id) {
      try {
        await loadCompareContext(active.session_id)
      } catch (err) {
        dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to restore default comparison' })
      }
    }
    toast('Baseline cleared.', { type: 'info' })
  }

  const dismissOnboarding = () => {
    setShowOnboarding(false)
    writeStorageItem(ONBOARDING_SEEN_KEY, 'true')
  }

  const saveNotes = async () => {
    if (!active?.session_id) return
    try {
      dispatch({ type: 'SAVE_NOTES_START' })
      await api.updateSessionNotes(active.session_id, noteDraft)
      dispatch({ type: 'SAVE_NOTES_COMPLETE' })
      toast('Notes saved.', { type: 'success' })
    } catch (err) {
      dispatch({ type: 'SAVE_NOTES_FAILED', error: err instanceof Error ? err.message : 'Failed to save notes' })
      toast('Failed to save notes.', { type: 'error' })
    }
  }

  const resetMainViewer = useCallback(() => setMainViewer({ scale: 1, x: 0, y: 0 }), [])
  const resetCompareViewer = useCallback(() => setCompareViewer({ scale: 1, x: 0, y: 0 }), [])
  const handleLesionHover = useCallback((key: string | null) => dispatch({ type: 'SET_ACTIVE_LESION', key }), [])
  const handleNoteDraftChange = useCallback((draft: string) => dispatch({ type: 'SET_NOTE_DRAFT', draft }), [])

  const toggleFullscreen = async () => {
    if (!workspaceRef.current) return
    if (!document.fullscreenElement) {
      await workspaceRef.current.requestFullscreen()
      setIsFullscreen(true)
    } else {
      await document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  // Stable callback refs for async handlers — avoids dependency chains
  const handleSelectHistoryRef = useRef(handleSelectHistory)
  handleSelectHistoryRef.current = handleSelectHistory
  const handlePurgeRef = useRef(handlePurge)
  handlePurgeRef.current = handlePurge
  const pinBaselineRef = useRef(pinBaseline)
  pinBaselineRef.current = pinBaseline
  const handleStartRef = useRef(handleStart)
  handleStartRef.current = handleStart
  const handleExportRef = useRef(handleExport)
  handleExportRef.current = handleExport
  const saveNotesRef = useRef(saveNotes)
  saveNotesRef.current = saveNotes
  const clearBaselineRef = useRef(clearBaseline)
  clearBaselineRef.current = clearBaseline
  const toggleFullscreenRef = useRef(toggleFullscreen)
  toggleFullscreenRef.current = toggleFullscreen

  const toggleDetectionOverlay = useCallback(() => setShowDetectionOverlay((v: boolean) => !v), [setShowDetectionOverlay])
  const toggleCompareOverlay = useCallback(() => setShowCompareOverlay((v: boolean) => !v), [setShowCompareOverlay])
  const handleSelectHistoryStable = useCallback((item: SessionSummary) => void handleSelectHistoryRef.current(item), [])
  const handlePurgeStable = useCallback((item: SessionSummary) => void handlePurgeRef.current(item), [])
  const handlePinBaselineStable = useCallback((item: SessionSummary) => void pinBaselineRef.current(item), [])
  const handleLoadMoreStable = useCallback(() => void loadMoreHistory(), [loadMoreHistory])
  const handleStartStable = useCallback(() => void handleStartRef.current(), [])
  const handleExportStable = useCallback(() => void handleExportRef.current(), [])
  const handleSaveNotesStable = useCallback(() => void saveNotesRef.current(), [])
  const handleClearBaselineStable = useCallback(() => void clearBaselineRef.current(), [])
  const handleToggleFullscreenStable = useCallback(() => void toggleFullscreenRef.current(), [])

  const handleProfileChange = (nextProfileId: string) => {
    setActiveProfileId(nextProfileId)
    if (active?.session_id && activeSessionProfileId !== nextProfileId) {
      dispatch({ type: 'SET_PROFILE_GUARD', guard: {
        sessionId: active.session_id,
        sessionProfileId: activeSessionProfileId,
        requestedProfileId: nextProfileId,
      } })
    } else {
      dispatch({ type: 'SET_PROFILE_GUARD', guard: null })
    }
  }

  useKeyboardShortcuts({
    active,
    compare,
    setMainViewer,
    resetMainViewer,
    resetCompareViewer,
    setShowDetectionOverlay,
    setCompareViewerMode,
    toggleFullscreen,
  })

  // ---------- Render ----------

  return (
    <section ref={workspaceRef} className="medical-grid relative bg-black py-32">
      <div className="mx-auto max-w-[1600px] px-8">
        <WorkspaceHeader
          activeProfileId={activeProfileId}
          activeProfileSummary={activeProfileSummary}
          retentionHours={retentionHours}
        />

        {profileSwitchNotice ? (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            role="status"
            aria-live="polite"
            className="pointer-events-none absolute right-8 top-10 z-20 rounded-2xl border border-cyan-400/15 bg-black/75 px-4 py-3 shadow-[0_0_30px_rgba(0,242,255,0.08)] backdrop-blur"
          >
            <div className="terminal-text text-[9px] tracking-[0.3em] text-cyan-400/80">PROFILE SWITCHED</div>
            <div className="mt-2 text-sm text-zinc-300">
              Workspace moved from <span className="text-zinc-500">{profileSwitchNotice.from}</span> to <span className="text-white">{profileSwitchNotice.to}</span>
            </div>
          </motion.div>
        ) : null}

        {profileGuard ? (
          <ProfileGuardBanner
            profileGuard={profileGuard}
            onReturnToProfile={() => {
              setActiveProfileId(profileGuard.sessionProfileId)
              dispatch({ type: 'SET_PROFILE_GUARD', guard: null })
            }}
            onResetCase={resetCaseWorkspace}
          />
        ) : null}

        <div className="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(260px,var(--left-rail))_minmax(0,1fr)_minmax(320px,var(--right-rail))]" style={{ ['--left-rail' as string]: `${leftRailWidth}px`, ['--right-rail' as string]: `${rightRailWidth}px` }}>
          <SessionArchiveRail
            activeProfileId={activeProfileId}
            onProfileChange={handleProfileChange}
            profiles={profiles}
            history={history}
            baselineSession={baselineSession}
            onSelectHistory={handleSelectHistoryStable}
            onPurge={handlePurgeStable}
            onPinBaseline={handlePinBaselineStable}
            leftRailWidth={leftRailWidth}
            onLeftRailWidthChange={setLeftRailWidth}
            hasMore={Boolean(historyCursor)}
            isLoadingMore={isLoadingMore}
            onLoadMore={handleLoadMoreStable}
          />

          <main className="holographic-panel rounded-[2rem] p-8">
            <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h3 className="terminal-text text-[10px] text-cyan-400/80">LIVE DIAGNOSTIC CANVAS</h3>
                <div className="text-sm text-zinc-500" aria-live="polite">
                  {status ? `${status.stage} · ${status.detail}` : 'Awaiting upload'}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 text-sm text-zinc-400">
                  <input type="checkbox" checked={privacyMode} onChange={(e) => setPrivacyMode(e.target.checked)} />
                  Privacy mode
                </label>
                <label className="inline-flex items-center gap-2 text-sm text-zinc-400">
                  Retention
                  <input
                    type="number"
                    min={1}
                    max={privacy?.max_retention_hours ?? 720}
                    value={retentionHours}
                    onChange={(e) => setRetentionHours(Number(e.target.value))}
                    className="w-20 rounded-lg border border-white/10 bg-black/40 px-2 py-1 text-sm text-white"
                  />
                </label>
              </div>
            </div>

            {isLoadingSession ? (
              <SessionLoadingSkeleton />
            ) : !active ? (
              <UploadPanel
                previewUrl={previewUrl}
                isAnalyzing={isAnalyzing}
                status={status}
                privacyMode={privacyMode}
                retentionHours={retentionHours}
                privacy={privacy}
                selectedFile={selectedFile}
                onFileChange={handleFileChange}
                onStart={handleStartStable}
              />
            ) : (
              <AnimatePresence mode="wait">
                <motion.div
                  key={active.session_id}
                  ref={resultsRef}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  transition={{ duration: 0.35, ease: 'easeOut' }}
                  className="space-y-8"
                >
                <div role="group" aria-label="Analysis summary metrics" className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                  <MetricCard label="SEVERITY" value={displaySeverity} accent tone={severityTone} />
                  <MetricCard label="GAGS SCORE" value={String(displayGags)} accent />
                  <MetricCard label="LESIONS" value={String(displayLesions)} />
                  <MetricCard label="SYMMETRY" value={`${displaySymmetry}%`} />
                </div>

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.7fr]">
                  <DetectionCanvas
                    activeImage={activeImage}
                    displayGags={displayGags}
                    displaySeverity={displaySeverity}
                    severityTone={severityTone}
                    showDetectionOverlay={showDetectionOverlay}
                    onToggleOverlay={toggleDetectionOverlay}
                    mainViewer={mainViewer}
                    onMainViewerChange={setMainViewer}
                    onResetMainViewer={resetMainViewer}
                    isFullscreen={isFullscreen}
                    onToggleFullscreen={handleToggleFullscreenStable}
                    lesionOverlayItems={lesionOverlayItems}
                    activeLesionKey={activeLesionKey}
                    onLesionHover={handleLesionHover}
                  />

                  <ClinicalGradePanel
                    active={active}
                    displayGags={displayGags}
                    displaySeverity={displaySeverity}
                    severityTone={severityTone}
                    regionRows={regionRows}
                    consensusLesions={consensusLesions}
                  />
                </div>

                {compare && (
                  <CompareSection
                    compareViewerMode={compareViewerMode}
                    compareFullscreen={compareFullscreen}
                    activeImage={activeImage}
                    previousImage={previousImage}
                    showCompareOverlay={showCompareOverlay}
                    onToggleCompareOverlay={toggleCompareOverlay}
                    compareViewer={compareViewer}
                    onCompareViewerChange={setCompareViewer}
                    onResetCompareViewer={resetCompareViewer}
                    onSetCompareViewerMode={setCompareViewerMode}
                    onSetCompareFullscreen={setCompareFullscreen}
                    lesionOverlayItems={lesionOverlayItems}
                    activeLesionKey={activeLesionKey}
                    onLesionHover={handleLesionHover}
                    activeDiagnosticImage={active.diagnostic_image ?? null}
                    activeOriginalImage={active.original_image ?? null}
                    previousDiagnosticImage={previousSession?.diagnostic_image ?? null}
                    previousOriginalImage={previousSession?.original_image ?? null}
                  />
                )}

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                  <CaseComparePanel
                    compare={compare}
                    baselineSession={baselineSession}
                    isPinnedBaselineCompare={isPinnedBaselineCompare}
                    isViewingPinnedBaseline={isViewingPinnedBaseline}
                    regionDeltaCards={regionDeltaCards}
                    onClearBaseline={handleClearBaselineStable}
                  />

                  <PrivacyExportPanel
                    privacyMode={privacyMode}
                    retentionHours={retentionHours}
                    sessionId={active.session_id}
                    exportPreset={exportPreset}
                    onExportPresetChange={setExportPreset}
                    onExport={handleExportStable}
                    onNewCase={resetCaseWorkspace}
                    isExporting={isExporting}
                  />
                </div>

                <LesionTable
                  lesions={lesionOverlayItems}
                  filteredLesions={filteredLesionOverlayItems}
                  lesionRegions={lesionRegions}
                  lesionTypes={lesionTypes}
                  lesionRegionFilter={lesionRegionFilter}
                  lesionConfidenceFilter={lesionConfidenceFilter}
                  lesionTypeFilter={lesionTypeFilter}
                  onRegionFilterChange={setLesionRegionFilter}
                  onConfidenceFilterChange={setLesionConfidenceFilter}
                  onTypeFilterChange={setLesionTypeFilter}
                  activeLesionKey={activeLesionKey}
                  onLesionHover={handleLesionHover}
                />

                <SessionNotesPanel
                  noteDraft={noteDraft}
                  onDraftChange={handleNoteDraftChange}
                  onSave={handleSaveNotesStable}
                  isSaving={isSavingNotes}
                />
              </motion.div>
              </AnimatePresence>
            )}

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  transition={{ duration: 0.25 }}
                  role="alert"
                  className="mt-6 flex items-start justify-between gap-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200"
                >
                  <span>{error}</span>
                  <button
                    type="button"
                    onClick={() => dispatch({ type: 'SET_ERROR', error: null })}
                    className="shrink-0 rounded-full border border-red-400/30 px-3 py-1 text-[10px] text-red-300 transition-colors hover:bg-red-400/10"
                  >
                    Dismiss
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {showOnboarding && <OnboardingModal onDismiss={dismissOnboarding} />}
          </main>

          <SystemStateRail
            status={status}
            rightRailWidth={rightRailWidth}
            onRightRailWidthChange={setRightRailWidth}
          />
        </div>
      </div>
    </section>
  )
}

// ---------- Private sub-components ----------

const WorkspaceHeader = memo(function WorkspaceHeader({
  activeProfileId,
  activeProfileSummary,
  retentionHours,
}: {
  activeProfileId: string
  activeProfileSummary: ProfileSummary | null
  retentionHours: number
}) {
  return (
    <div className="mb-16 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <p className="terminal-text mb-4 text-[10px] tracking-[0.35em] text-cyan-400/70">CLINICAL WORKSPACE</p>
        <h2 className="text-5xl font-bold tracking-tighter md:text-6xl">Neural Archive Workspace</h2>
        <p className="mt-4 max-w-2xl text-lg text-zinc-500">
          Longitudinal diagnosis, consensus inspection, privacy controls, and case comparison in a single clinical shell.
        </p>
        <div className="mt-5 flex flex-wrap items-center gap-3 text-xs">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/8 px-3 py-2 text-cyan-100">
            <span className="terminal-text text-[9px] tracking-[0.24em] text-cyan-400/80">ACTIVE PROFILE</span>
            <span className="font-medium text-white">{activeProfileId}</span>
          </div>
          {activeProfileSummary ? (
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-zinc-300">
              <span>{activeProfileSummary.sessions} archived sessions</span>
              <span className="text-zinc-500">latest {activeProfileSummary.latest_severity ?? 'pending'}</span>
            </div>
          ) : null}
        </div>
      </div>

      <div className="holographic-panel flex items-center gap-4 rounded-2xl px-6 py-4">
        <Shield aria-hidden="true" className="h-5 w-5 text-cyan-400" />
        <div>
          <div className="terminal-text text-[9px] text-cyan-400/80">RETENTION WINDOW</div>
          <div className="text-sm text-zinc-300">{retentionHours} hours</div>
        </div>
      </div>
    </div>
  )
})

function ProfileGuardBanner({
  profileGuard,
  onReturnToProfile,
  onResetCase,
}: {
  profileGuard: {
    sessionId: string
    sessionProfileId: string
    requestedProfileId: string
  }
  onReturnToProfile: () => void
  onResetCase: () => void
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      role="alert"
      className="mb-6 flex flex-col gap-4 rounded-[1.5rem] border border-amber-400/20 bg-amber-400/8 px-5 py-4 text-sm text-amber-50 shadow-[0_0_30px_rgba(251,191,36,0.08)] lg:flex-row lg:items-center lg:justify-between"
    >
      <div>
        <div className="terminal-text text-[9px] tracking-[0.28em] text-amber-200/80">ACTIVE CASE PROFILE MISMATCH</div>
        <p className="mt-2 max-w-3xl text-amber-50/90">
          Session {profileGuard.sessionId.slice(0, 8)} belongs to profile {profileGuard.sessionProfileId}. You switched the archive to {profileGuard.requestedProfileId}. Reset the current case or return to the matching profile archive.
        </p>
      </div>
      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onReturnToProfile}
          className="rounded-full border border-amber-200/30 px-4 py-2 text-[11px] font-medium text-amber-50 hover:bg-amber-200/10"
        >
          Return to {profileGuard.sessionProfileId}
        </button>
        <button
          type="button"
          onClick={onResetCase}
          className="rounded-full bg-amber-300 px-4 py-2 text-[11px] font-semibold text-black"
        >
          Reset active case
        </button>
      </div>
    </motion.div>
  )
}
