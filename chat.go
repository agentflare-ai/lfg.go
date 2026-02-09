//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"unsafe"
)

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
	Role    string
	Content string
}

// ApplyChatTemplate formats chat messages using a template.
// If tmpl is empty, the model's default template is used (requires a model).
// If addAssistant is true, the output ends with the assistant prompt prefix.
func ApplyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	registerChatFuncs()

	// Build cChatMessage array with Go byte slices for strings.
	cMessages := make([]cChatMessage, len(messages))
	keepAlive := make([][]byte, 0, len(messages)*2)

	for i, msg := range messages {
		roleBytes := cString(msg.Role)
		contentBytes := cString(msg.Content)
		keepAlive = append(keepAlive, roleBytes, contentBytes)
		cMessages[i].Role = cStringPtr(roleBytes)
		cMessages[i].Content = cStringPtr(contentBytes)
	}

	var tmplPtr uintptr
	var tmplBytes []byte
	if tmpl != "" {
		tmplBytes = cString(tmpl)
		tmplPtr = cStringPtr(tmplBytes)
	}

	var msgPtr uintptr
	if len(cMessages) > 0 {
		msgPtr = uintptr(unsafe.Pointer(&cMessages[0]))
	}

	// First pass: determine required size.
	n := _lfg_chat_apply_template(tmplPtr, msgPtr, uintptr(len(messages)), addAssistant, 0, 0)
	runtime.KeepAlive(keepAlive)
	runtime.KeepAlive(tmplBytes)
	if n <= 0 {
		return "", &Error{Code: ErrorInternal, Message: "failed to apply chat template"}
	}

	buf := make([]byte, int(n)+1) // +1 for null terminator space
	n = _lfg_chat_apply_template(tmplPtr, msgPtr, uintptr(len(messages)), addAssistant, uintptr(unsafe.Pointer(&buf[0])), int32(len(buf)))
	runtime.KeepAlive(keepAlive)
	runtime.KeepAlive(tmplBytes)
	runtime.KeepAlive(cMessages)
	if n <= 0 {
		return "", &Error{Code: ErrorInternal, Message: "failed to apply chat template"}
	}
	return string(buf[:n]), nil
}

// ChatBuiltinTemplates returns a list of built-in chat template names.
func ChatBuiltinTemplates() []string {
	registerChatFuncs()
	// Get count first with a small buffer.
	var buf [64]uintptr
	n := _lfg_chat_builtin_templates(uintptr(unsafe.Pointer(&buf[0])), 64)
	if n <= 0 {
		return nil
	}
	result := make([]string, n)
	for i := 0; i < int(n); i++ {
		result[i] = goString(buf[i])
	}
	return result
}
