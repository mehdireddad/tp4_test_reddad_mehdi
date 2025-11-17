package ma.emsi.reddad;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.reddad.llm.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
        import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;

public class TestAIorNOT {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {

        configureLogger();

        System.out.println("=== Phase 1 : Ingestion du document RAG ===");

        // 1️⃣ Chargement du PDF sur le RAG
        DocumentParser parser = new ApacheTikaDocumentParser();
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        // 2️⃣ Découpage en segments
        var splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments générés : " + segments.size());

        // 3️⃣ Embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // 4️⃣ Store
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        System.out.println("✅ Embeddings ajoutés au store mémoire.");

        // 5️⃣ Modèle Gemini
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null) throw new IllegalStateException("❌ GEMINI_KEY manquante !");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 6️⃣ Retriever
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println("\n=== Phase 2 : Routage intelligent (RAG ou pas) ===");

        // 7️⃣ QueryRouter personnalisé
        class RoutageIntelligent implements QueryRouter {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                // 🔹 Création du template
                PromptTemplate template = PromptTemplate.from(
                        "Est-ce que la requête suivante concerne l'IA, le RAG (Retrieval-Augmented Generation) ou le Fine-Tuning ? " +
                                "Réponds uniquement par 'oui', 'non' ou 'peut-être'.\n\nRequête : {{question}}"
                );

                // 🔹 Application du template
                var prompt = template.apply(Map.of("question", query.text()));

                // 🔹 Envoi direct au modèle
                String reponse = model.chat(prompt.text()).trim().toLowerCase();
                System.out.println("🧭 Décision du modèle : " + reponse);

                // 🔹 Routage conditionnel
                if (reponse.contains("non")) {
                    System.out.println("🚫 Pas de RAG utilisé pour cette question.");
                    return Collections.emptyList();
                } else {
                    System.out.println("✅ RAG activé (contexte du PDF utilisé).");
                    return List.of(retriever);
                }
            }
        }

        QueryRouter queryRouter = new RoutageIntelligent();

        // 8️⃣ Création du RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 9️⃣ Création de l’assistant
        var memory = MessageWindowChatMemory.withMaxMessages(10);
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(memory)
                .build();

        System.out.println("\n=== Assistant prêt ===");
        System.out.println("💬 Tapez vos questions (ou 'bye' pour quitter)");

        // 🔟 Interaction utilisateur
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n👤 Vous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("bye")) break;

                // 🧠 Réponse du modèle
                String reponse = assistant.chat(question);

                // Si pas de RAG, on donne une réponse générale
                if (reponse == null || reponse.isBlank()) {
                    System.out.println("🤖 (Pas de RAG) Réponse sans contexte :");
                    reponse = model.chat("Réponds naturellement à cette question sans utiliser de document externe : " + question);
                }

                System.out.println("🤖 Gemini : " + reponse);
            }
        }
    }
}