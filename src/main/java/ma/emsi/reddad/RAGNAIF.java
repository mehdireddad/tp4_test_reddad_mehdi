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

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class RAGNAIF {

    public static void main(String[] args) throws Exception {
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



    }
}