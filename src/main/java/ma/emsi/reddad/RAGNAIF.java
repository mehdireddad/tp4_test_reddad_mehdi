package ma.emsi.reddad;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import ma.emsi.reddad.llm.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;



public class RAGNAIF {
    private static void configureLogger() {
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


        // Phase 1 : Embeddings
        // Création du parser PDF (Apache Tika)
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        // Chargement du fichier PDF
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);

        // Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments : " + segments.size());

        // Création du modèle d’embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Génération des embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre d'embeddings générés : " + embeddings.size());

        // Création du magasin d’embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Ajout des embeddings et segments associés
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Enregistrement des embeddings terminé avec succès !");

        String cle = System.getenv("GEMINI_KEY");
        if (cle == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY manquante !!!");
        }

        // Création du modèle de chat Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(cle)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        // Création du ContentRetriever
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Ajout d'une mémoire de 10 messages
        var memory = MessageWindowChatMemory.withMaxMessages(10);

        // Création de l’assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        //  Interaction console (multi-questions)
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Posez votre question :");
            while (true) {
                System.out.print("(Tapez 'bye' pour quitter) Vous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("bye"))
                    break;
                String reponse = assistant.chat(question);
                System.out.println("Gemini : " + reponse);
            }
        }


    }
}