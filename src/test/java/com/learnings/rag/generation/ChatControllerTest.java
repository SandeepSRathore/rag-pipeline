package com.learnings.rag.generation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.http.MediaType.TEXT_EVENT_STREAM;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import reactor.core.publisher.Flux;

@WebMvcTest(ChatController.class)
class ChatControllerTest {

    @Autowired
    MockMvc mvc;

    @MockitoBean
    AnswerService answerService;

    @Test
    void streamsEventsAsNamedServerSentEvents() throws Exception {
        when(answerService.answer("What is HNSW?")).thenReturn(Flux.just(
                new ChatEvent.Sources(List.of(new SourceRef(1, "pgvector.adoc", "PGvector", "Indexes", 0.8, "text"))),
                new ChatEvent.Token("HNSW [1]"),
                new ChatEvent.Done(10, 2, 5, 40)));

        MvcResult started = mvc.perform(post("/api/chat")
                        .contentType(APPLICATION_JSON)
                        .accept(TEXT_EVENT_STREAM)
                        .content("{\"question\":\"  What is HNSW?  \"}"))
                .andExpect(request().asyncStarted())
                .andReturn();

        String body = mvc.perform(asyncDispatch(started))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(TEXT_EVENT_STREAM))
                .andReturn().getResponse().getContentAsString();

        assertThat(body).containsSubsequence(
                "event:sources", "\"sourcePath\":\"pgvector.adoc\"",
                "event:token", "\"text\":\"HNSW [1]\"",
                "event:done", "\"promptTokens\":10");
    }

    @Test
    void blankQuestionIsRejected() throws Exception {
        mvc.perform(post("/api/chat").contentType(APPLICATION_JSON).content("{\"question\":\"   \"}"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void invalidQuestionIsStill400WhenTheClientAsksForAnEventStream() throws Exception {
        // The UI always sends Accept: text/event-stream; a validation error must not turn into a 406 or 500.
        mvc.perform(post("/api/chat").contentType(APPLICATION_JSON).accept(TEXT_EVENT_STREAM)
                        .content("{\"question\":\"   \"}"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void overlongQuestionIsRejected() throws Exception {
        String question = "x".repeat(2001);
        mvc.perform(post("/api/chat").contentType(APPLICATION_JSON).content("{\"question\":\"" + question + "\"}"))
                .andExpect(status().isBadRequest());
    }
}
