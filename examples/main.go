package main

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"

	lfg "github.com/agentflare-ai/lfg.go"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const systemPrompt = `You are a helpful AI assistant. Answer clearly and concisely.
When uncertain, ask clarifying questions. You have automatic memory:
confident statements are remembered, and relevant context is injected
when you're uncertain.`

// ---------------------------------------------------------------------------
// TUI message types
// ---------------------------------------------------------------------------

type chatEntry struct {
	role string // "user", "assistant", "system", "stored"
	text string
}

type (
	tokenMsg        string
	streamClearMsg  struct{}
	responseMsg     string // assistant's plain text response
	generationDone  struct{ err error }
	memRetrievalMsg struct {
		text  string
		count int
	}
	memStoredMsg struct {
		text       string
		spanLength int32
	}
	memSurpriseMsg struct {
		inputText        string
		meanSurprise     float32
		nAboveThreshold  int32
		nTokensEvaluated int32
	}
	statusMsg   string
	errMsg      struct{ err error }
	modelLoaded struct {
		model           *lfg.Model
		session         *lfg.Session
		memory          *vectorMemory
		nEntropyEmbd    int32
		nConfidenceEmbd int32
		nSurpriseEmbd   int32
	}
)

// ---------------------------------------------------------------------------
// Vector Memory Store
// ---------------------------------------------------------------------------

type memoryEntry struct {
	text      string
	embedding []float32
	source    string // "confidence" | "user"
}

type vectorMemory struct {
	mu      sync.Mutex
	entries []memoryEntry
	session *lfg.Session
}

func newVectorMemory(s *lfg.Session) *vectorMemory {
	return &vectorMemory{session: s}
}

func (vm *vectorMemory) storeWithEmbedding(text string, embedding []float32, source string) {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	vm.entries = append(vm.entries, memoryEntry{text: text, embedding: embedding, source: source})
}

func (vm *vectorMemory) store(text, source string) {
	emb, err := vm.session.Embed(text)
	if err != nil {
		return
	}
	vm.storeWithEmbedding(text, emb, source)
}

func (vm *vectorMemory) search(embedding []float32, topK int) []string {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	if len(vm.entries) == 0 {
		return nil
	}

	type scored struct {
		text  string
		score float64
	}

	results := make([]scored, 0, len(vm.entries))
	for _, e := range vm.entries {
		sim := cosineSimilarity(embedding, e.embedding)
		results = append(results, scored{text: e.text, score: sim})
	}

	slices.SortFunc(results, func(a, b scored) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return 0
	})

	if topK > len(results) {
		topK = len(results)
	}

	out := make([]string, topK)
	for i := range topK {
		out[i] = results[i].text
	}
	return out
}

func (vm *vectorMemory) searchByText(text string, topK int) []string {
	emb, err := vm.session.Embed(text)
	if err != nil {
		return nil
	}
	return vm.search(emb, topK)
}

func (vm *vectorMemory) count() int {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	return len(vm.entries)
}

func (vm *vectorMemory) autoCount() int {
	vm.mu.Lock()
	defer vm.mu.Unlock()
	n := 0
	for _, e := range vm.entries {
		if e.source == "confidence" {
			n++
		}
	}
	return n
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// ---------------------------------------------------------------------------
// Token Accumulator (tracks generated pieces for span text extraction)
// ---------------------------------------------------------------------------

type tokenAccumulator struct {
	mu     sync.Mutex
	pieces []string
}

func (a *tokenAccumulator) add(piece string) {
	a.mu.Lock()
	a.pieces = append(a.pieces, piece)
	a.mu.Unlock()
}

func (a *tokenAccumulator) extractLastN(n int) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	total := len(a.pieces)
	start := total - n
	if start < 0 {
		start = 0
	}
	var sb strings.Builder
	for _, p := range a.pieces[start:total] {
		sb.WriteString(p)
	}
	return sb.String()
}

func (a *tokenAccumulator) text() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	var sb strings.Builder
	for _, p := range a.pieces {
		sb.WriteString(p)
	}
	return sb.String()
}

type entropyMonitorEvent struct {
	event     lfg.EntropyEvent
	embedding []float32
}

type confidenceMonitorEvent struct {
	event     lfg.ConfidenceEvent
	embedding []float32
}

type surpriseMonitorEvent struct {
	event     lfg.SurpriseEvent
	embedding []float32
}

func cloneEmbedding(buf []float32, n int32) []float32 {
	if n <= 0 {
		return nil
	}
	size := int(n)
	if size > len(buf) {
		size = len(buf)
	}
	if size <= 0 {
		return nil
	}
	out := make([]float32, size)
	copy(out, buf[:size])
	return out
}

func startEntropyPump(session *lfg.Session, nEmbd int32, done <-chan struct{}, out chan<- entropyMonitorEvent, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(out)

		counter := session.EntropyCounter()
		var seen int32
		if counter != nil {
			seen = atomic.LoadInt32(counter)
		}

		buf := make([]float32, max(int(nEmbd), 1))
		popAll := func() bool {
			for {
				event, ok := session.EntropyPop(buf)
				if !ok {
					return false
				}
				select {
				case out <- entropyMonitorEvent{event: event, embedding: cloneEmbedding(buf, event.NEmbedding)}:
				case <-done:
					return true
				}
			}
		}

		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-done:
				popAll()
				return
			case <-ticker.C:
				if counter != nil {
					cur := atomic.LoadInt32(counter)
					if cur == seen {
						continue
					}
					seen = cur
				}
				if popAll() {
					return
				}
			}
		}
	}()
}

func startConfidencePump(session *lfg.Session, nEmbd int32, done <-chan struct{}, out chan<- confidenceMonitorEvent, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(out)

		counter := session.ConfidenceCounter()
		var seen int32
		if counter != nil {
			seen = atomic.LoadInt32(counter)
		}

		buf := make([]float32, max(int(nEmbd), 1))
		popAll := func() bool {
			for {
				event, ok := session.ConfidencePop(buf)
				if !ok {
					return false
				}
				select {
				case out <- confidenceMonitorEvent{event: event, embedding: cloneEmbedding(buf, event.NEmbedding)}:
				case <-done:
					return true
				}
			}
		}

		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-done:
				popAll()
				return
			case <-ticker.C:
				if counter != nil {
					cur := atomic.LoadInt32(counter)
					if cur == seen {
						continue
					}
					seen = cur
				}
				if popAll() {
					return
				}
			}
		}
	}()
}

func startSurprisePump(session *lfg.Session, nEmbd int32, done <-chan struct{}, out chan<- surpriseMonitorEvent, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(out)

		counter := session.SurpriseCounter()
		var seen int32
		if counter != nil {
			seen = atomic.LoadInt32(counter)
		}

		buf := make([]float32, max(int(nEmbd), 1))
		popAll := func() bool {
			for {
				event, ok := session.SurprisePop(buf)
				if !ok {
					return false
				}
				select {
				case out <- surpriseMonitorEvent{event: event, embedding: cloneEmbedding(buf, event.NEmbedding)}:
				case <-done:
					return true
				}
			}
		}

		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-done:
				popAll()
				return
			case <-ticker.C:
				if counter != nil {
					cur := atomic.LoadInt32(counter)
					if cur == seen {
						continue
					}
					seen = cur
				}
				if popAll() {
					return
				}
			}
		}
	}()
}

// ---------------------------------------------------------------------------
// Agent Turn Logic
// ---------------------------------------------------------------------------

func runAgentTurn(
	session *lfg.Session,
	memory *vectorMemory,
	history []lfg.ChatMessage,
	userText string,
	eventCh chan<- tea.Msg,
	cfg agentConfig,
) {
	defer func() { eventCh <- generationDone{} }()

	// Store user message in memory for future retrieval.
	memory.store(userText, "user")

	// Build message array: system + history + user.
	messages := make([]lfg.ChatMessage, 0, len(history)+3)
	messages = append(messages, lfg.ChatMessage{Role: "system", Content: systemPrompt})
	messages = append(messages, history...)

	// Pre-retrieve memory for the incoming user message.
	if memory.count() > 0 {
		if retrieved := memory.searchByText(userText, 3); len(retrieved) > 0 {
			contextText := strings.Join(retrieved, "\n")
			messages = append(messages, lfg.ChatMessage{
				Role:    "system",
				Content: "Relevant context:\n" + contextText,
			})
			eventCh <- memRetrievalMsg{text: contextText, count: len(retrieved)}
		}
	}

	messages = append(messages, lfg.ChatMessage{Role: "user", Content: userText})

	accum := &tokenAccumulator{}

	session.Reset()
	session.EntropyFlush()
	session.ConfidenceFlush()
	session.SurpriseFlush()

	entropyEvents := make(chan entropyMonitorEvent, 64)
	confidenceEvents := make(chan confidenceMonitorEvent, 64)
	surpriseEvents := make(chan surpriseMonitorEvent, 32)
	stopPumps := make(chan struct{})
	var pumpWG sync.WaitGroup
	startEntropyPump(session, cfg.entropyNEmbd, stopPumps, entropyEvents, &pumpWG)
	startConfidencePump(session, cfg.confidenceNEmbd, stopPumps, confidenceEvents, &pumpWG)
	startSurprisePump(session, cfg.surpriseNEmbd, stopPumps, surpriseEvents, &pumpWG)

	genCfg := lfg.GenerateConfig{
		MaxTokens: cfg.maxTokens,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			accum.add(piece)
			eventCh <- tokenMsg(piece)
			return lfg.GenerateContinue
		},
	}

	processEntropy := func(e entropyMonitorEvent) {
		if len(e.embedding) == 0 || memory.count() == 0 {
			return
		}
		results := memory.search(e.embedding, 3)
		if len(results) == 0 {
			return
		}
		eventCh <- memRetrievalMsg{
			text:  strings.Join(results, "\n"),
			count: len(results),
		}
	}

	processConfidence := func(e confidenceMonitorEvent) {
		if len(e.embedding) == 0 {
			return
		}
		storeText := stripThinking(e.event.SpanText)
		if storeText == "" {
			return
		}
		memory.storeWithEmbedding(storeText, append([]float32(nil), e.embedding...), "confidence")
		eventCh <- memStoredMsg{text: storeText, spanLength: e.event.SpanLength}
	}

	processSurprise := func(e surpriseMonitorEvent) {
		if len(e.embedding) > 0 && userText != "" {
			memory.storeWithEmbedding(userText, append([]float32(nil), e.embedding...), "surprise")
		}
		eventCh <- memSurpriseMsg{
			inputText:        userText,
			meanSurprise:     e.event.MeanSurprise,
			nAboveThreshold:  e.event.NAboveThreshold,
			nTokensEvaluated: e.event.NTokensEvaluated,
		}
	}

	type generateResult struct {
		err error
	}
	genDone := make(chan generateResult, 1)
	go func() {
		_, err := session.ChatGenerate(messages, genCfg)
		genDone <- generateResult{err: err}
	}()

	var genErr error
running:
	for {
		select {
		case e := <-entropyEvents:
			processEntropy(e)
		case e := <-confidenceEvents:
			processConfidence(e)
		case e := <-surpriseEvents:
			processSurprise(e)
		case done := <-genDone:
			genErr = done.err
			break running
		}
	}

	close(stopPumps)
	pumpWG.Wait()

	for e := range entropyEvents {
		processEntropy(e)
	}
	for e := range confidenceEvents {
		processConfidence(e)
	}
	for e := range surpriseEvents {
		processSurprise(e)
	}

	if genErr != nil {
		eventCh <- statusMsg(fmt.Sprintf("Error: %v", genErr))
		return
	}

	output := stripThinking(accum.text())
	eventCh <- responseMsg(output)
}

// ---------------------------------------------------------------------------
// Agent configuration
// ---------------------------------------------------------------------------

type agentConfig struct {
	modelPath           string
	threads             int
	ctxSize             int
	temperature         float32
	entropyThreshold    float32
	entropyNEmbd        int32
	confidenceThreshold float32
	confidenceMinSpan   int32
	confidenceNEmbd     int32
	surpriseThreshold   float32
	surpriseNEmbd       int32
	maxTokens           int32
	gpuLayers           int
	reasoningBudget     int
}

// ---------------------------------------------------------------------------
// Bubbletea Model
// ---------------------------------------------------------------------------

type model struct {
	input         textinput.Model
	messages      []chatEntry
	streaming     string
	status        string
	err           error
	session       *lfg.Session
	lfgModel      *lfg.Model
	memory        *vectorMemory
	loaded        bool
	generating    bool
	eventCh       chan tea.Msg
	pendingUser   string
	history       []lfg.ChatMessage
	quitting      bool
	config        agentConfig
	activeTab     int            // 0 = Chat, 1 = Memory
	memoryEntries []chatEntry    // stored/retrieval events
	memViewport   viewport.Model // scrollable viewport for memory tab
	width, height int            // terminal dimensions
	thinkingDone  bool           // true once </think> seen in current stream
}

func initialModel(cfg agentConfig) model {
	ti := textinput.New()
	ti.Placeholder = "Type a message..."
	ti.Focus()
	ti.CharLimit = 4096

	return model{
		input:       ti,
		config:      cfg,
		eventCh:     make(chan tea.Msg, 256),
		status:      "Loading model...",
		memViewport: viewport.New(80, 20),
	}
}

// TestableModel creates a model for testing without a real LFG model.
func TestableModel() model {
	ti := textinput.New()
	ti.Placeholder = "Type a message..."
	ti.Focus()

	return model{
		input:       ti,
		eventCh:     make(chan tea.Msg, 256),
		status:      "Test Mode",
		memViewport: viewport.New(80, 20),
	}
}

func (m model) Init() tea.Cmd {
	return m.loadModelCmd()
}

func (m model) loadModelCmd() tea.Cmd {
	cfg := m.config
	return func() tea.Msg {
		mdl, err := lfg.LoadModelSimple(cfg.modelPath, lfg.WithGPULayers(cfg.gpuLayers))
		if err != nil {
			return errMsg{err: fmt.Errorf("load model: %w", err)}
		}

		session, err := lfg.NewSession(mdl,
			lfg.WithSessionNCtx(cfg.ctxSize),
			lfg.WithSessionThreads(cfg.threads),
			lfg.WithSessionMaxTokens(cfg.maxTokens),
			lfg.WithSessionHealing(true),
			lfg.WithSessionStructuredCheckpointing(true),
			lfg.WithSessionReasoningBudget(cfg.reasoningBudget),
			lfg.WithSessionSampling(lfg.SamplingConfig{
				Temp: cfg.temperature,
				TopK: 40,
				TopP: 0.9,
				MinP: 0.05,
			}),
		)
		if err != nil {
			mdl.Close()
			return errMsg{err: fmt.Errorf("create session: %w", err)}
		}

		// Entropy monitor — async retrieval signal queue.
		nEntropyEmbd, err := session.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
			Threshold:      cfg.entropyThreshold,
			CooldownTokens: 5,
			RingSize:       8,
		})
		if err != nil {
			session.Close()
			mdl.Close()
			return errMsg{err: fmt.Errorf("configure entropy monitor: %w", err)}
		}

		// Confidence monitor — async confident-span storage queue.
		nConfidenceEmbd, err := session.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
			Threshold: cfg.confidenceThreshold,
			MinSpan:   cfg.confidenceMinSpan,
			RingSize:  8,
		})
		if err != nil {
			session.Close()
			mdl.Close()
			return errMsg{err: fmt.Errorf("configure confidence monitor: %w", err)}
		}

		// Surprise monitor — async prompt novelty queue.
		nSurpriseEmbd, err := session.ConfigureSurpriseMonitor(&lfg.SurpriseMonitorConfig{
			Threshold: cfg.surpriseThreshold,
			RingSize:  8,
		})
		if err != nil {
			session.Close()
			mdl.Close()
			return errMsg{err: fmt.Errorf("configure surprise monitor: %w", err)}
		}

		// Reasoning tokens for thinking support.
		vocab := mdl.Vocab()
		startTokens, _ := vocab.Tokenize("<think>", false, true)
		endTokens, _ := vocab.Tokenize("</think>", false, true)
		session.ConfigureReasoning(startTokens, endTokens)

		// Stop string so generation halts at end-of-turn.
		// Text-based matching is encoding-independent (works regardless
		// of how the tokenizer splits the text).
		session.ConfigureStopStrings([]string{"<|im_end|>"})

		mem := newVectorMemory(session)

		return modelLoaded{
			model:           mdl,
			session:         session,
			memory:          mem,
			nEntropyEmbd:    nEntropyEmbd,
			nConfidenceEmbd: nConfidenceEmbd,
			nSurpriseEmbd:   nSurpriseEmbd,
		}
	}
}

func waitForEvent(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		return <-ch
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			m.quitting = true
			return m, tea.Quit
		case tea.KeyTab:
			m.activeTab = (m.activeTab + 1) % 2
			return m, nil
		case tea.KeyEnter:
			if m.activeTab != 0 {
				return m, nil
			}
			if !m.generating && m.loaded {
				text := strings.TrimSpace(m.input.Value())
				if text != "" {
					m.messages = append(m.messages, chatEntry{role: "user", text: text})
					m.pendingUser = text
					m.input.Reset()
					m.generating = true
					m.streaming = ""
					m.thinkingDone = false
					m.status = "Generating..."

					historyCopy := make([]lfg.ChatMessage, len(m.history))
					copy(historyCopy, m.history)

					go runAgentTurn(m.session, m.memory, historyCopy, text, m.eventCh, m.config)
					return m, waitForEvent(m.eventCh)
				}
			}
		}

		// On memory tab, forward keys to viewport for scrolling.
		if m.activeTab == 1 {
			var cmd tea.Cmd
			m.memViewport, cmd = m.memViewport.Update(msg)
			return m, cmd
		}

		// Pass keys to text input when not generating.
		if !m.generating {
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			return m, cmd
		}
		return m, nil

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.Width = msg.Width - 4
		// header(1) + blank(1) + tab bar(1) + blank(1) = 4 top lines
		// input(1) + status(1) + blank(1) = 3 bottom lines
		vpHeight := msg.Height - 7
		if vpHeight < 1 {
			vpHeight = 1
		}
		m.memViewport.Width = msg.Width
		m.memViewport.Height = vpHeight
		return m, nil

	case modelLoaded:
		m.lfgModel = msg.model
		m.session = msg.session
		m.memory = msg.memory
		m.config.entropyNEmbd = msg.nEntropyEmbd
		m.config.confidenceNEmbd = msg.nConfidenceEmbd
		m.config.surpriseNEmbd = msg.nSurpriseEmbd
		m.loaded = true
		m.status = "Ready"
		return m, nil

	case tokenMsg:
		m.streaming += string(msg)
		if !m.thinkingDone && strings.Contains(m.streaming, "</think>") {
			m.thinkingDone = true
		}
		return m, waitForEvent(m.eventCh)

	case streamClearMsg:
		m.streaming = ""
		return m, waitForEvent(m.eventCh)

	case responseMsg:
		text := string(msg)
		m.messages = append(m.messages, chatEntry{role: "assistant", text: text})
		m.history = append(m.history,
			lfg.ChatMessage{Role: "user", Content: m.pendingUser},
			lfg.ChatMessage{Role: "assistant", Content: text},
		)
		m.streaming = ""
		return m, waitForEvent(m.eventCh)

	case memRetrievalMsg:
		m.memoryEntries = append(m.memoryEntries, chatEntry{
			role: "system",
			text: fmt.Sprintf("[recall] Retrieved %d items:\n%s", msg.count, msg.text),
		})
		m.memViewport.SetContent(renderMemoryContent(m.memoryEntries, m.memViewport.Width))
		m.memViewport.GotoBottom()
		return m, waitForEvent(m.eventCh)

	case memStoredMsg:
		m.memoryEntries = append(m.memoryEntries, chatEntry{
			role: "stored",
			text: fmt.Sprintf("[stored] %q (%dt)", truncate(msg.text, 40), msg.spanLength),
		})
		m.memViewport.SetContent(renderMemoryContent(m.memoryEntries, m.memViewport.Width))
		m.memViewport.GotoBottom()
		return m, waitForEvent(m.eventCh)

	case memSurpriseMsg:
		m.memoryEntries = append(m.memoryEntries, chatEntry{
			role: "system",
			text: fmt.Sprintf("[surprise] Novel input (%.2f, %d/%d tokens): %s",
				msg.meanSurprise, msg.nAboveThreshold, msg.nTokensEvaluated, msg.inputText),
		})
		m.memViewport.SetContent(renderMemoryContent(m.memoryEntries, m.memViewport.Width))
		m.memViewport.GotoBottom()
		return m, waitForEvent(m.eventCh)

	case statusMsg:
		m.status = string(msg)
		return m, waitForEvent(m.eventCh)

	case generationDone:
		m.generating = false
		if msg.err != nil {
			m.err = msg.err
			m.status = fmt.Sprintf("Error: %v", msg.err)
		} else {
			total, auto := 0, 0
			if m.memory != nil {
				total = m.memory.count()
				auto = m.memory.autoCount()
			}
			if total > 0 {
				m.status = fmt.Sprintf("Ready | %d memories (%d auto)", total, auto)
			} else {
				m.status = "Ready"
			}
		}
		return m, nil

	case errMsg:
		m.err = msg.err
		m.status = fmt.Sprintf("Error: %v", msg.err)
		return m, nil
	}

	return m, nil
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

var (
	headerStyle      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	userStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
	assistantStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	systemStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("243"))
	storedStyle      = lipgloss.NewStyle().Faint(true).Foreground(lipgloss.Color("51"))
	streamStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	statusStyle      = lipgloss.NewStyle().Faint(true)
	dimStyle         = lipgloss.NewStyle().Faint(true)
	activeTabStyle   = lipgloss.NewStyle().Bold(true).Underline(true)
	inactiveTabStyle = lipgloss.NewStyle().Faint(true)
)

func renderMemoryContent(entries []chatEntry, width int) string {
	if len(entries) == 0 {
		return "No memory events yet."
	}
	if width <= 0 {
		width = 80
	}
	wrap := lipgloss.NewStyle().Width(width)
	var sb strings.Builder
	for _, entry := range entries {
		switch entry.role {
		case "system":
			sb.WriteString(wrap.Render(systemStyle.Render(entry.text)))
		case "stored":
			sb.WriteString(wrap.Render(storedStyle.Render(entry.text)))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func (m model) View() string {
	if m.quitting {
		return ""
	}

	var sb strings.Builder

	sb.WriteString(headerStyle.Render("LFG Chat Agent"))
	sb.WriteString("\n")

	// Tab bar.
	chatLabel := "Chat"
	memLabel := "Memory"
	if n := len(m.memoryEntries); n > 0 && m.activeTab == 0 {
		memLabel = fmt.Sprintf("Memory (%d)", n)
	}
	if m.activeTab == 0 {
		sb.WriteString(activeTabStyle.Render("["+chatLabel+"]") + "  " + inactiveTabStyle.Render(memLabel))
	} else {
		sb.WriteString(inactiveTabStyle.Render(chatLabel) + "  " + activeTabStyle.Render("["+memLabel+"]"))
	}
	sb.WriteString("\n\n")

	if m.activeTab == 0 {
		m.renderChatTab(&sb)
	} else {
		sb.WriteString(m.memViewport.View())
		sb.WriteString("\n")
	}

	sb.WriteString("\n")

	if !m.loaded && m.err == nil {
		sb.WriteString(dimStyle.Render("Loading model..."))
	} else if m.generating {
		sb.WriteString(dimStyle.Render("(generating — press Ctrl+C to quit)"))
	} else if m.activeTab == 0 {
		sb.WriteString(m.input.View())
	} else {
		sb.WriteString(dimStyle.Render("(Tab to switch back to chat)"))
	}

	sb.WriteString("\n")
	sb.WriteString(statusStyle.Render(m.status))

	return sb.String()
}

func (m model) renderChatTab(sb *strings.Builder) {
	for _, entry := range m.messages {
		switch entry.role {
		case "user":
			sb.WriteString(userStyle.Render("You: " + entry.text))
		case "assistant":
			sb.WriteString(assistantStyle.Render("Agent: " + entry.text))
		}
		sb.WriteString("\n")
	}

	if m.streaming != "" {
		display := stripThinking(m.streaming)
		if display != "" && m.thinkingDone {
			sb.WriteString(streamStyle.Render("Agent: " + display))
		} else {
			sb.WriteString(dimStyle.Render("Thinking..."))
		}
		sb.WriteString("\n")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// stripThinking removes thinking blocks from generated text.
// Handles three cases:
//  1. Normal:  <think>...</think>Response   — strips the block.
//  2. Leading: reasoning...</think>Response  — chat template added <think> to prompt,
//     so generated text starts mid-thinking. Strips everything before </think>.
//  3. Unclosed: text<think>...               — partial stream, discard after <think>.
func stripThinking(s string) string {
	var sb strings.Builder
	for {
		end := strings.Index(s, "</think>")
		if end < 0 {
			// No </think>. Check for unclosed <think> at end.
			if start := strings.Index(s, "<think>"); start >= 0 {
				sb.WriteString(s[:start])
			} else {
				sb.WriteString(s)
			}
			break
		}
		// Found </think>. Check for matching <think> before it.
		start := strings.Index(s[:end], "<think>")
		if start >= 0 {
			// Normal case: keep text before <think>.
			sb.WriteString(s[:start])
		}
		// Leading case (no <think>): skip everything before </think>.
		s = s[end+len("</think>"):]
	}
	return strings.TrimSpace(sb.String())
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// ---------------------------------------------------------------------------
// Cobra CLI
// ---------------------------------------------------------------------------

func main() {
	var cfg agentConfig

	rootCmd := &cobra.Command{
		Use:   "lfg-chat",
		Short: "Interactive chat agent powered by lfg.go",
		Long: `An interactive chat agent that showcases lfg.go features:
  - Plain text chat with automatic memory
  - Entropy-triggered memory retrieval and mid-stream context injection
  - Confidence-triggered memory storage (automatic)
  - Surprise-triggered novel input detection
  - Thinking/reasoning support`,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			m := initialModel(cfg)
			p := tea.NewProgram(m, tea.WithAltScreen())
			_, err := p.Run()
			return err
		},
	}

	f := rootCmd.Flags()
	f.StringVarP(&cfg.modelPath, "model", "m",
		"./models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
		"Path to GGUF model file")
	f.IntVarP(&cfg.threads, "threads", "t", 4, "Number of threads")
	f.IntVarP(&cfg.ctxSize, "context", "c", 2048, "Context size")
	f.Float32VarP(&cfg.temperature, "temperature", "T", 0.7, "Sampling temperature")
	f.Float32VarP(&cfg.entropyThreshold, "entropy-threshold", "e", 0.6,
		"Entropy threshold for memory retrieval")
	f.Float32Var(&cfg.confidenceThreshold, "confidence-threshold", 0.3,
		"Entropy ceiling for confident tokens")
	f.Int32Var(&cfg.confidenceMinSpan, "confidence-min-span", 5,
		"Min consecutive confident tokens to auto-store")
	f.Float32Var(&cfg.surpriseThreshold, "surprise-threshold", 0.5,
		"Surprise threshold for novel input detection")
	f.Int32VarP(&cfg.maxTokens, "max-tokens", "n", 512, "Max tokens per generation")
	f.IntVarP(&cfg.gpuLayers, "gpu-layers", "g", 0, "Number of GPU layers")
	f.IntVar(&cfg.reasoningBudget, "reasoning-budget", 256, "Reasoning token budget")

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
