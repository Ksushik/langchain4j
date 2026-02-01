package dev.langchain4j.memory.chat;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureGreaterThanZero;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.memory.ChatMemoryService;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

/**
 * A chat memory implementation that uses summarization to compress older messages when the token limit is exceeded.
 * <p>
 * This memory operates as a hybrid approach: it keeps recent messages verbatim and summarizes older ones
 * when the total token count exceeds the configured maximum. The summary is stored as a system message
 * that provides context from the earlier conversation.
 * <p>
 * Similar to LangChain Python's {@code ConversationSummaryBufferMemory}, this implementation:
 * <ul>
 * <li>Maintains a "moving summary" of older conversation turns</li>
 * <li>Keeps recent messages intact until they need to be summarized</li>
 * <li>Uses progressive summarization to incorporate new messages into the existing summary</li>
 * </ul>
 * <p>
 * The rules for {@link SystemMessage}:
 * <ul>
 * <li>The original system message (if any) is always retained at position 0</li>
 * <li>The summary message is stored separately and prepended to the messages list when retrieved</li>
 * <li>Only one original {@code SystemMessage} can be held at a time</li>
 * </ul>
 * <p>
 * If an {@link AiMessage} containing {@link ToolExecutionRequest}(s) would be summarized,
 * the following orphan {@link ToolExecutionResultMessage}(s) are also included in the summary
 * to avoid issues with some LLM providers.
 * <p>
 * The state of chat memory is stored in {@link ChatMemoryStore} ({@link SingleSlotChatMemoryStore} is used by default).
 *
 * @see MessageWindowChatMemory
 * @see TokenWindowChatMemory
 */
public class SummaryChatMemory implements ChatMemory {

    /**
     * Default prompt template used for progressive summarization.
     */
    public static final String DEFAULT_SUMMARY_PROMPT_TEMPLATE =
            """
            Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

            EXAMPLE
            Current summary:
            The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

            New lines of conversation:
            Human: Why do you think artificial intelligence is a force for good?
            AI: Because artificial intelligence will help humans reach their full potential.

            New summary:
            The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
            END OF EXAMPLE

            Current summary:
            %s

            New lines of conversation:
            %s

            New summary:
            """;

    private static final String SUMMARY_PREFIX = "[Conversation Summary] ";

    private final Object id;
    private final Function<Object, Integer> maxTokensProvider;
    private final TokenCountEstimator tokenCountEstimator;
    private final ChatModel summarizationModel;
    private final String summaryPromptTemplate;
    private final ChatMemoryStore store;

    // Summary is stored separately from regular messages
    private String currentSummary = "";

    private SummaryChatMemory(Builder builder) {
        this.id = ensureNotNull(builder.id, "id");
        this.maxTokensProvider = ensureNotNull(builder.maxTokensProvider, "maxTokensProvider");
        ensureGreaterThanZero(this.maxTokensProvider.apply(this.id), "maxTokens");
        this.tokenCountEstimator = ensureNotNull(builder.tokenCountEstimator, "tokenCountEstimator");
        this.summarizationModel = ensureNotNull(builder.summarizationModel, "summarizationModel");
        this.summaryPromptTemplate = getOrDefault(builder.summaryPromptTemplate, DEFAULT_SUMMARY_PROMPT_TEMPLATE);
        this.store = ensureNotNull(builder.store(), "store");
    }

    @Override
    public Object id() {
        return id;
    }

    @Override
    public void add(ChatMessage message) {
        List<ChatMessage> messages = new LinkedList<>(store.getMessages(id));

        if (message instanceof SystemMessage) {
            Optional<SystemMessage> existingSystemMessage = SystemMessage.findFirst(messages);
            if (existingSystemMessage.isPresent()) {
                if (existingSystemMessage.get().equals(message)) {
                    return; // do not add the same system message
                } else {
                    messages.remove(existingSystemMessage.get()); // need to replace existing system message
                }
            }
            // System message always goes at position 0
            messages.add(0, message);
        } else {
            messages.add(message);
        }

        Integer maxTokens = maxTokensProvider.apply(id);
        ensureGreaterThanZero(maxTokens, "maxTokens");
        ensureCapacityWithSummarization(messages, maxTokens);

        store.updateMessages(id, messages);
    }

    @Override
    public void set(Iterable<ChatMessage> iter) {
        if (iter instanceof List) {
            set((List<ChatMessage>) iter);
        } else {
            List<ChatMessage> list = new ArrayList<>();
            iter.forEach(list::add);
            set(list);
        }
    }

    private void set(List<ChatMessage> messages) {
        Integer maxTokens = maxTokensProvider.apply(id);
        ensureGreaterThanZero(maxTokens, "maxTokens");

        List<ChatMessage> mutableMessages = new LinkedList<>(messages);
        ensureCapacityWithSummarization(mutableMessages, maxTokens);

        store.updateMessages(id, mutableMessages);
    }

    @Override
    public List<ChatMessage> messages() {
        Integer maxTokens = maxTokensProvider.apply(id);
        ensureGreaterThanZero(maxTokens, "maxTokens");

        List<ChatMessage> storedMessages = new LinkedList<>(store.getMessages(id));
        ensureCapacityWithSummarization(storedMessages, maxTokens);

        // Build the final list with summary as a system message if present
        List<ChatMessage> result = new ArrayList<>();

        if (!currentSummary.isEmpty()) {
            result.add(SystemMessage.from(SUMMARY_PREFIX + currentSummary));
        }

        result.addAll(storedMessages);

        return result;
    }

    /**
     * Returns the current summary of the conversation.
     *
     * @return the current summary, or an empty string if no summarization has occurred
     */
    public String summary() {
        return currentSummary;
    }

    @Override
    public void clear() {
        currentSummary = "";
        store.deleteMessages(id);
    }

    private void ensureCapacityWithSummarization(List<ChatMessage> messages, int maxTokens) {
        if (messages.isEmpty()) {
            return;
        }

        int currentTokenCount = estimateTokenCount(messages);

        while (currentTokenCount > maxTokens && messages.size() > 1) {
            // Find messages to summarize (oldest non-system messages)
            List<ChatMessage> messagesToSummarize = new ArrayList<>();

            int messageToSummarizeIndex = 0;
            // Skip system message if present at position 0
            if (!messages.isEmpty() && messages.get(0) instanceof SystemMessage) {
                messageToSummarizeIndex = 1;
            }

            if (messageToSummarizeIndex >= messages.size()) {
                // Only system message left, nothing to summarize
                return;
            }

            // Extract the oldest message
            ChatMessage messageToSummarize = messages.remove(messageToSummarizeIndex);
            messagesToSummarize.add(messageToSummarize);

            // If it's an AiMessage with tool execution requests, also include orphan ToolExecutionResultMessages
            if (messageToSummarize instanceof AiMessage aiMessage && aiMessage.hasToolExecutionRequests()) {
                while (messageToSummarizeIndex < messages.size()
                        && messages.get(messageToSummarizeIndex) instanceof ToolExecutionResultMessage) {
                    messagesToSummarize.add(messages.remove(messageToSummarizeIndex));
                }
            }

            // Update the summary with the messages being evicted
            currentSummary = summarize(messagesToSummarize, currentSummary);

            // Recalculate token count
            currentTokenCount = estimateTokenCount(messages);
        }
    }

    private int estimateTokenCount(List<ChatMessage> messages) {
        int messageTokens = tokenCountEstimator.estimateTokenCountInMessages(messages);
        // Also account for the summary if present
        if (!currentSummary.isEmpty()) {
            messageTokens += tokenCountEstimator.estimateTokenCountInText(SUMMARY_PREFIX + currentSummary);
        }
        return messageTokens;
    }

    private String summarize(List<ChatMessage> messagesToSummarize, String existingSummary) {
        String newLines = formatMessagesForSummary(messagesToSummarize);
        String previousSummary = existingSummary.isEmpty() ? "No previous summary." : existingSummary;

        String prompt = String.format(summaryPromptTemplate, previousSummary, newLines);

        return summarizationModel.chat(prompt);
    }

    private String formatMessagesForSummary(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage message : messages) {
            if (message instanceof UserMessage userMessage) {
                sb.append("Human: ").append(extractText(userMessage)).append("\n");
            } else if (message instanceof AiMessage aiMessage) {
                if (aiMessage.text() != null && !aiMessage.text().isEmpty()) {
                    sb.append("AI: ").append(aiMessage.text()).append("\n");
                }
                if (aiMessage.hasToolExecutionRequests()) {
                    for (ToolExecutionRequest request : aiMessage.toolExecutionRequests()) {
                        sb.append("AI: [Called tool: ").append(request.name()).append("]\n");
                    }
                }
            } else if (message instanceof ToolExecutionResultMessage toolResult) {
                sb.append("Tool Result: ").append(toolResult.text()).append("\n");
            }
            // Skip SystemMessage in summary - it's handled separately
        }
        return sb.toString().trim();
    }

    private String extractText(UserMessage userMessage) {
        if (userMessage.hasSingleText()) {
            return userMessage.singleText();
        }
        // For multi-content messages, concatenate text contents
        StringBuilder sb = new StringBuilder();
        userMessage.contents().forEach(content -> {
            if (content instanceof dev.langchain4j.data.message.TextContent textContent) {
                sb.append(textContent.text()).append(" ");
            }
        });
        return sb.toString().trim();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private Object id = ChatMemoryService.DEFAULT;
        private Function<Object, Integer> maxTokensProvider;
        private TokenCountEstimator tokenCountEstimator;
        private ChatModel summarizationModel;
        private String summaryPromptTemplate;
        private ChatMemoryStore store;

        /**
         * Sets the ID of the {@link ChatMemory}.
         * If not provided, "default" will be used.
         *
         * @param id The ID of the ChatMemory
         * @return this builder
         */
        public Builder id(Object id) {
            this.id = id;
            return this;
        }

        /**
         * Sets the maximum number of tokens to retain before summarization is triggered.
         * When the total token count exceeds this limit, older messages are summarized.
         *
         * @param maxTokens            The maximum number of tokens
         * @param tokenCountEstimator  A {@link TokenCountEstimator} for counting tokens in messages
         * @return this builder
         */
        public Builder maxTokens(Integer maxTokens, TokenCountEstimator tokenCountEstimator) {
            this.maxTokensProvider = (id) -> maxTokens;
            this.tokenCountEstimator = tokenCountEstimator;
            return this;
        }

        /**
         * Sets a dynamic maximum number of tokens provider.
         * The value returned by this provider may change dynamically at runtime.
         *
         * @param maxTokensProvider    A provider that returns the maximum number of tokens
         * @param tokenCountEstimator  A {@link TokenCountEstimator} for counting tokens in messages
         * @return this builder
         */
        public Builder dynamicMaxTokens(
                Function<Object, Integer> maxTokensProvider, TokenCountEstimator tokenCountEstimator) {
            this.maxTokensProvider = maxTokensProvider;
            this.tokenCountEstimator = tokenCountEstimator;
            return this;
        }

        /**
         * Sets the chat model used for generating summaries.
         * This model is called whenever older messages need to be summarized.
         *
         * @param summarizationModel The {@link ChatModel} to use for summarization
         * @return this builder
         */
        public Builder summarizationModel(ChatModel summarizationModel) {
            this.summarizationModel = summarizationModel;
            return this;
        }

        /**
         * Sets a custom prompt template for summarization.
         * The template should contain two {@code %s} placeholders:
         * <ol>
         * <li>For the current/previous summary</li>
         * <li>For the new lines of conversation to be summarized</li>
         * </ol>
         * If not provided, {@link #DEFAULT_SUMMARY_PROMPT_TEMPLATE} will be used.
         *
         * @param summaryPromptTemplate The custom prompt template
         * @return this builder
         */
        public Builder summaryPromptTemplate(String summaryPromptTemplate) {
            this.summaryPromptTemplate = summaryPromptTemplate;
            return this;
        }

        /**
         * Sets the chat memory store responsible for storing the chat memory state.
         * If not provided, an in-memory {@link SingleSlotChatMemoryStore} will be used.
         *
         * @param store The {@link ChatMemoryStore} to use
         * @return this builder
         */
        public Builder chatMemoryStore(ChatMemoryStore store) {
            this.store = store;
            return this;
        }

        private ChatMemoryStore store() {
            return store != null ? store : new SingleSlotChatMemoryStore(id);
        }

        /**
         * Builds the {@link SummaryChatMemory} instance.
         *
         * @return a new SummaryChatMemory instance
         */
        public SummaryChatMemory build() {
            return new SummaryChatMemory(this);
        }
    }

    /**
     * Creates a new SummaryChatMemory with the specified configuration.
     *
     * @param maxTokens            The maximum number of tokens before summarization is triggered
     * @param tokenCountEstimator  The estimator for counting tokens
     * @param summarizationModel   The model to use for generating summaries
     * @return a new SummaryChatMemory instance
     */
    public static SummaryChatMemory withMaxTokens(
            int maxTokens, TokenCountEstimator tokenCountEstimator, ChatModel summarizationModel) {
        return builder()
                .maxTokens(maxTokens, tokenCountEstimator)
                .summarizationModel(summarizationModel)
                .build();
    }
}
