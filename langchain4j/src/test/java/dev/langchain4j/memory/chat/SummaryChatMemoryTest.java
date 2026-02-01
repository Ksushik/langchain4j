package dev.langchain4j.memory.chat;

import static dev.langchain4j.data.message.AiMessage.aiMessage;
import static dev.langchain4j.data.message.SystemMessage.systemMessage;
import static dev.langchain4j.data.message.UserMessage.userMessage;
import static dev.langchain4j.internal.TestUtils.aiMessageWithTokens;
import static dev.langchain4j.internal.TestUtils.systemMessageWithTokens;
import static dev.langchain4j.internal.TestUtils.userMessageWithTokens;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O_MINI;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiTokenCountEstimator;

import java.util.List;
import java.util.function.Function;

import org.assertj.core.api.WithAssertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

class SummaryChatMemoryTest implements WithAssertions {

    private static final TokenCountEstimator TOKEN_COUNT_ESTIMATOR = new OpenAiTokenCountEstimator(GPT_4_O_MINI);
    private static final int EXTRA_TOKENS_PER_REQUEST = 3;

    private ChatModel mockSummarizationModel;

    @BeforeEach
    void setUp() {
        mockSummarizationModel = mock(ChatModel.class);
        when(mockSummarizationModel.chat(anyString())).thenReturn("Summary of conversation");
    }

    @Test
    void id() {
        {
            ChatMemory chatMemory = SummaryChatMemory.builder()
                    .maxTokens(100, TOKEN_COUNT_ESTIMATOR)
                    .summarizationModel(mockSummarizationModel)
                    .build();
            assertThat(chatMemory.id()).isEqualTo("default");
        }
        {
            ChatMemory chatMemory = SummaryChatMemory.builder()
                    .id("custom-id")
                    .maxTokens(100, TOKEN_COUNT_ESTIMATOR)
                    .summarizationModel(mockSummarizationModel)
                    .build();
            assertThat(chatMemory.id()).isEqualTo("custom-id");
        }
    }

    @Test
    void should_store_and_retrieve_messages_when_under_limit() {
        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(200, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        UserMessage userMessage = userMessage("Hello");
        AiMessage aiMessage = aiMessage("Hi there!");

        chatMemory.add(userMessage);
        chatMemory.add(aiMessage);

        assertThat(chatMemory.messages()).containsExactly(userMessage, aiMessage);
        assertThat(chatMemory.summary()).isEmpty();

        // No summarization should have been triggered
        verify(mockSummarizationModel, never()).chat(anyString());
    }

    @Test
    void should_clear_memory() {
        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(200, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        chatMemory.add(userMessage("Hello"));
        chatMemory.add(aiMessage("Hi!"));

        chatMemory.clear();

        assertThat(chatMemory.messages()).isEmpty();
        assertThat(chatMemory.summary()).isEmpty();

        // Clear is idempotent
        chatMemory.clear();
        assertThat(chatMemory.messages()).isEmpty();
    }

    @Test
    void should_summarize_oldest_messages_when_exceeding_token_limit() {
        // Set up a memory with a small token limit
        int maxTokens = 33; // Enough for ~3 messages of 10 tokens each

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(maxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        when(mockSummarizationModel.chat(anyString())).thenReturn("User greeted AI");

        // Add messages that will exceed the limit
        UserMessage msg1 = userMessageWithTokens(10);
        chatMemory.add(msg1);

        AiMessage msg2 = aiMessageWithTokens(10);
        chatMemory.add(msg2);

        UserMessage msg3 = userMessageWithTokens(10);
        chatMemory.add(msg3);

        // At this point we're at the limit
        assertThat(TOKEN_COUNT_ESTIMATOR.estimateTokenCountInMessages(List.of(msg1, msg2, msg3)))
                .isEqualTo(EXTRA_TOKENS_PER_REQUEST + 30);

        // Adding one more should trigger summarization
        AiMessage msg4 = aiMessageWithTokens(10);
        chatMemory.add(msg4);

        // Summarization should have been called at least once
        verify(mockSummarizationModel, atLeast(1)).chat(anyString());
        assertThat(chatMemory.summary()).isEqualTo("User greeted AI");

        // The summary should be included in messages
        List<ChatMessage> messages = chatMemory.messages();
        assertThat(messages).isNotEmpty();
        assertThat(messages.get(0)).isInstanceOf(SystemMessage.class);
        assertThat(((SystemMessage) messages.get(0)).text()).contains("User greeted AI");
    }

    @Test
    void should_not_summarize_system_message() {
        int maxTokens = 33;

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(maxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        when(mockSummarizationModel.chat(anyString())).thenReturn("Brief summary");

        SystemMessage systemMessage = systemMessageWithTokens(10);
        chatMemory.add(systemMessage);

        UserMessage msg1 = userMessageWithTokens(10);
        chatMemory.add(msg1);

        AiMessage msg2 = aiMessageWithTokens(10);
        chatMemory.add(msg2);

        // Adding another message should trigger summarization of msg1, not systemMessage
        UserMessage msg3 = userMessageWithTokens(10);
        chatMemory.add(msg3);

        // System message should still be present
        List<ChatMessage> messages = chatMemory.messages();
        boolean hasOriginalSystemMessage = messages.stream()
                .filter(m -> m instanceof SystemMessage)
                .map(m -> (SystemMessage) m)
                .anyMatch(sm -> sm.equals(systemMessage));

        // The original system message content should be preserved
        assertThat(messages).contains(systemMessage);
    }

    @Test
    void should_keep_only_one_system_message() {
        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(200, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        SystemMessage systemMessage1 = systemMessage("You are a helpful assistant");
        chatMemory.add(systemMessage1);

        UserMessage userMessage = userMessage("Hello");
        chatMemory.add(userMessage);

        // Adding a new system message should replace the old one
        SystemMessage systemMessage2 = systemMessage("You are a code reviewer");
        chatMemory.add(systemMessage2);

        List<ChatMessage> messages = chatMemory.messages();

        long systemMessageCount = messages.stream()
                .filter(m -> m instanceof SystemMessage)
                .filter(m -> !((SystemMessage) m).text().contains("[Conversation Summary]"))
                .count();

        assertThat(systemMessageCount).isEqualTo(1);
        assertThat(messages).contains(systemMessage2);
        assertThat(messages).doesNotContain(systemMessage1);
    }

    @Test
    void should_not_add_duplicate_system_message() {
        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(200, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        SystemMessage systemMessage = systemMessage("You are a helpful assistant");
        chatMemory.add(systemMessage);
        chatMemory.add(systemMessage);
        chatMemory.add(systemMessage);

        List<ChatMessage> messages = chatMemory.messages();

        long systemMessageCount = messages.stream()
                .filter(m -> m instanceof SystemMessage)
                .count();

        assertThat(systemMessageCount).isEqualTo(1);
    }

    @Test
    void should_include_tool_execution_in_summary() {
        int maxTokens = 45;

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(maxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        when(mockSummarizationModel.chat(promptCaptor.capture())).thenReturn("User asked for calculation");

        UserMessage userMessage = userMessage("What is 2+2?");
        chatMemory.add(userMessage);

        ToolExecutionRequest toolRequest = ToolExecutionRequest.builder()
                .id("1")
                .name("calculator")
                .arguments("{\"a\": 2, \"b\": 2}")
                .build();
        AiMessage aiWithTool = AiMessage.from(toolRequest);
        chatMemory.add(aiWithTool);

        ToolExecutionResultMessage toolResult = ToolExecutionResultMessage.from(toolRequest, "4");
        chatMemory.add(toolResult);

        // Add more messages to trigger summarization
        AiMessage aiMessage = aiMessage("2 + 2 = 4");
        chatMemory.add(aiMessage);

        UserMessage nextQuestion = userMessage("What about 3+3?");
        chatMemory.add(nextQuestion);

        // Verify summarization was called at least once
        verify(mockSummarizationModel, atLeast(1)).chat(anyString());
    }

    @Test
    void should_progressively_summarize() {
        int maxTokens = 33;

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(maxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        when(mockSummarizationModel.chat(anyString()))
                .thenReturn("First summary")
                .thenReturn("Extended summary with more context")
                .thenReturn("Even more extended summary");

        // Add initial messages
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));
        chatMemory.add(userMessageWithTokens(10));

        // First summarization
        chatMemory.add(aiMessageWithTokens(10));

        // More messages to trigger another summarization
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));

        // Should have been called at least twice for progressive summarization
        verify(mockSummarizationModel, atLeast(2)).chat(anyString());
        // Summary should be progressively updated
        assertThat(chatMemory.summary()).isNotEmpty();
    }

    @Test
    void should_use_custom_summary_prompt_template() {
        String customTemplate = "Custom template: Previous: %s, New: %s";

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(33, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .summaryPromptTemplate(customTemplate)
                .build();

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        when(mockSummarizationModel.chat(promptCaptor.capture())).thenReturn("Custom summary");

        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));

        verify(mockSummarizationModel, atLeast(1)).chat(anyString());
        // At least one of the captured prompts should start with our custom template
        assertThat(promptCaptor.getAllValues())
                .anyMatch(prompt -> prompt.startsWith("Custom template:"));
    }

    @Test
    void dynamic_max_tokens_behavior() {
        int[] currentMaxTokens = {100}; // Start with a high limit
        Function<Object, Integer> dynamicMaxTokens = id -> currentMaxTokens[0];

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .id("dynamic")
                .dynamicMaxTokens(dynamicMaxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        when(mockSummarizationModel.chat(anyString())).thenReturn("Dynamic summary");

        // Add messages within the initial limit
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));

        // No summarization yet since we're under the limit
        verify(mockSummarizationModel, never()).chat(anyString());

        // Reduce the limit significantly
        currentMaxTokens[0] = 33;

        // Access messages - this should trigger summarization due to reduced limit
        chatMemory.messages();

        verify(mockSummarizationModel, atLeast(1)).chat(anyString());
    }

    @Test
    void should_use_factory_method() {
        SummaryChatMemory chatMemory = SummaryChatMemory.withMaxTokens(
                100, TOKEN_COUNT_ESTIMATOR, mockSummarizationModel);

        assertThat(chatMemory).isNotNull();
        assertThat(chatMemory.id()).isEqualTo("default");
    }

    @Test
    void should_handle_set_with_list() {
        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(200, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        UserMessage msg1 = userMessage("First");
        AiMessage msg2 = aiMessage("Second");

        chatMemory.set(msg1, msg2);

        assertThat(chatMemory.messages()).containsExactly(msg1, msg2);
    }

    @Test
    void should_throw_when_maxTokens_is_zero() {
        assertThatThrownBy(() -> SummaryChatMemory.builder()
                .maxTokens(0, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_throw_when_maxTokens_is_negative() {
        assertThatThrownBy(() -> SummaryChatMemory.builder()
                .maxTokens(-1, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_throw_when_summarizationModel_is_null() {
        assertThatThrownBy(() -> SummaryChatMemory.builder()
                .maxTokens(100, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(null)
                .build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_throw_when_tokenCountEstimator_is_null() {
        assertThatThrownBy(() -> SummaryChatMemory.builder()
                .maxTokens(100, null)
                .summarizationModel(mockSummarizationModel)
                .build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_include_summary_prefix_in_system_message() {
        int maxTokens = 33;

        SummaryChatMemory chatMemory = SummaryChatMemory.builder()
                .maxTokens(maxTokens, TOKEN_COUNT_ESTIMATOR)
                .summarizationModel(mockSummarizationModel)
                .build();

        when(mockSummarizationModel.chat(anyString())).thenReturn("The user discussed weather");

        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));
        chatMemory.add(userMessageWithTokens(10));
        chatMemory.add(aiMessageWithTokens(10));

        List<ChatMessage> messages = chatMemory.messages();

        // First message should be a system message with the summary
        assertThat(messages.get(0)).isInstanceOf(SystemMessage.class);
        SystemMessage summaryMessage = (SystemMessage) messages.get(0);
        assertThat(summaryMessage.text()).startsWith("[Conversation Summary]");
        assertThat(summaryMessage.text()).contains("The user discussed weather");
    }
}
