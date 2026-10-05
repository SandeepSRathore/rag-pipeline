package com.learnings.rag.generation;

import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;
import reactor.core.publisher.Flux;

@RestController
public class ChatController {

    private final AnswerService answerService;

    public ChatController(AnswerService answerService) {
        this.answerService = answerService;
    }

    @PostMapping(path = "/api/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<ChatEvent>> chat(@Valid @RequestBody ChatRequest request) {
        return answerService.answer(request.question().strip())
                .map(event -> ServerSentEvent.<ChatEvent>builder(event).event(eventName(event)).build());
    }

    private static String eventName(ChatEvent event) {
        return switch (event) {
            case ChatEvent.Sources _ -> "sources";
            case ChatEvent.Token _ -> "token";
            case ChatEvent.Done _ -> "done";
            case ChatEvent.Error _ -> "error";
        };
    }
}
