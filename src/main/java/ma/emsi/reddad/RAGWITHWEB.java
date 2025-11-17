package ma.emsi.reddad;

import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.reddad.llm.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RAGWITHWEB {

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();

        // ====================== PDF & Embeddings ======================
        DocumentParser parser = new ApacheTikaDocumentParser();
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // ====================== Modèle Gemini ======================
        String GEMINI_KEY = System.getenv("GEMINI_API_KEY");
        if (GEMINI_KEY == null) throw new IllegalStateException("❌ GEMINI_KEY manquante !");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ====================== ContentRetrievers ======================
        EmbeddingStoreContentRetriever retrieverLocal = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        String TAVILY_KEY = System.getenv("TAVILY_KEY");
        if (TAVILY_KEY == null) throw new IllegalStateException("❌ Variable d'environnement TAVILY_KEY manquante !");
        TavilyWebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(TAVILY_KEY)
                .build();

        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilyEngine)
                .maxResults(3)
                .build();

        // ====================== QueryRouter & RetrievalAugmentor ======================
        QueryRouter router = new DefaultQueryRouter(List.of(retrieverLocal, retrieverWeb));
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // ====================== Assistant ======================
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // ====================== Interaction ======================
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n👤 Vous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("bye")) break;
                String response = assistant.chat(question);
                System.out.println("🤖 Gemini : " + response);
            }
        }
    }
}