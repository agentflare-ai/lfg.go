package lfg

/*
#include "lfg_inference.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
	Role    string
	Content string
}

// ApplyChatTemplate formats chat messages using a template.
// If tmpl is empty, the model's default template is used (requires a model).
// If addAssistant is true, the output ends with the assistant prompt prefix.
func ApplyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	cMessages := make([]C.struct_lfg_chat_message, len(messages))
	cStrings := make([]*C.char, 0, len(messages)*2)
	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	for i, msg := range messages {
		cRole := C.CString(msg.Role)
		cContent := C.CString(msg.Content)
		cStrings = append(cStrings, cRole, cContent)
		cMessages[i].role = cRole
		cMessages[i].content = cContent
	}

	var cTmpl *C.char
	if tmpl != "" {
		cTmpl = C.CString(tmpl)
		defer C.free(unsafe.Pointer(cTmpl))
	}

	var msgPtr *C.struct_lfg_chat_message
	if len(cMessages) > 0 {
		msgPtr = &cMessages[0]
	}

	// First pass: determine required size.
	n := C.lfg_chat_apply_template(cTmpl, msgPtr, C.size_t(len(messages)), C.bool(addAssistant), nil, 0)
	if n <= 0 {
		return "", &Error{Code: ErrorInternal, Message: "failed to apply chat template"}
	}

	buf := make([]byte, int(n)+1) // +1 for null terminator space
	n = C.lfg_chat_apply_template(cTmpl, msgPtr, C.size_t(len(messages)), C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n <= 0 {
		return "", &Error{Code: ErrorInternal, Message: "failed to apply chat template"}
	}
	return string(buf[:n]), nil
}

// ChatBuiltinTemplates returns a list of built-in chat template names.
func ChatBuiltinTemplates() []string {
	// Get count first with a small buffer.
	var buf [64]*C.char
	n := C.lfg_chat_builtin_templates(&buf[0], 64)
	if n <= 0 {
		return nil
	}
	result := make([]string, n)
	for i := 0; i < int(n); i++ {
		result[i] = C.GoString(buf[i])
	}
	return result
}
