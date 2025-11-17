package ma.emsi.reddad;


import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import ma.emsi.reddad.llm.Assistant;


import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
        import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.Handler;

public class TestOfRoutage {

    private static void configureLogger() {
        System.out.println("Configuring logger");
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) throws Exception {
        configureLogger();

        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Créer les magasins et récupérer les segments
        EmbeddingStoreWithSegments storeIAWithSegments = creerEmbeddingStore("src/main/resources/rag.pdf", embeddingModel, parser);
        EmbeddingStoreWithSegments storemlWithSegments = creerEmbeddingStore("src/main/resources/ml.pdf", embeddingModel, parser);

        EmbeddingStore<TextSegment> storeIA = storeIAWithSegments.store();
        EmbeddingStore<TextSegment> storeCuisine = storemlWithSegments.store();

        // Créer les retrievers
        var retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        var retrieverCuisine = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCuisine)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // Modèle Gemini
        String cle = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        // Routage
        Map<ContentRetriever, String> desc = new HashMap<>();
        desc.put(retrieverIA, "Documents de cours sur le RAG, le fine-tuning et l'intelligence artificielle");
        desc.put(retrieverCuisine, "Articles sur la cuisine, la gastronomie et les recettes");

        var queryRouter = new LanguageModelQueryRouter(model, desc);

        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("(Tapez 'bye' pour quitter) Vous : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("bye")) break;
            String reponse = assistant.chat(question);
            System.out.println("Gemini : " + reponse);
        }
    }

    // Classe interne pour renvoyer à la fois le store et les segments
    private static class EmbeddingStoreWithSegments {
        private final EmbeddingStore<TextSegment> store;
        private final List<TextSegment> segments;

        public EmbeddingStoreWithSegments(EmbeddingStore<TextSegment> store, List<TextSegment> segments) {
            this.store = store;
            this.segments = segments;
        }

        public EmbeddingStore<TextSegment> store() { return store; }
        public List<TextSegment> segments() { return segments; }
    }

    private static EmbeddingStoreWithSegments creerEmbeddingStore(String chemin, EmbeddingModel embeddingModel, DocumentParser parser) throws Exception {
        Path path = Paths.get(chemin);
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println("✅ Fichier chargé : " + chemin + " (" + segments.size() + " segments)");
        return new EmbeddingStoreWithSegments(store, segments);
    }

}