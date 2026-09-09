# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from aphrodite.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse, ChatCompletionStreamResponse


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"


class OmniChatCompletionResponse(ChatCompletionResponse):
    pass
