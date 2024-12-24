import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import path from "node:path";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

/**
 * @description Initializes the PdfQA class with the provided parameters for setting up the document querying pipeline.
 * @param model {string}: The name or path of the model to be used for processing the queries.
 * @param pdfDocument {string}: The path to the PDF document that will be loaded for querying.
 * @param chunkSize {number}: The maximum size of each text chunk when splitting the document.
 * @param chunkOverlap {number}: The number of overlapping tokens between consecutive chunks for context preservation.
 * @param searchType {string}: The algorithm to be used for searching (e.g., "similarity" for nearest neighbor search).
 * @param kDocuments {number}: The number of top documents to retrieve during the query process.
 */

class PdfQA {
  constructor({
    model,
    pdfDocument,
    chunkSize,
    chunkOverlap,
    searchType = "similarity",
    kDocuments,
  }) {
    this.model = model;
    this.pdfDocument = pdfDocument;
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;

    this.searchType = searchType;
    this.kDocuments = kDocuments;
  }

  async init() {
    this.initChatModel();
    await this.loadDocuments();
    await this.splitDocuments();
    this.selectEmbedding = new OllamaEmbeddings({ model: "all-minilm:latest" });
    await this.createVectorStore();
    this.createRetriever();
    this.chain = await this.createChain();
    return this;
  }

  /**
   * @description Initialize the chat model.
   * @returns void
   */
  initChatModel() {
    console.log("Loading model...");
    this.llm = new Ollama({ model: this.model });
  }

  /**
   * @description Load documents from a PDF file and convert to a format that can be ingested by the langchain document splitter.
   */
  async loadDocuments() {
    console.log("Loading PDFs...");
    const pdfLoader = new PDFLoader(
      path.join(import.meta.dirname, this.pdfDocument)
    );
    this.documents = await pdfLoader.load();
  }

  /**
   * @description Split the documents into chunks of a given size with a specified overlap.
   */
  async splitDocuments() {
    console.log("Splitting documents...");
    const textSplitter = new CharacterTextSplitter({
      separator: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });
    this.texts = await textSplitter.splitDocuments(this.documents);
  }

  /**
   * @description Create Vector Store.
   */
  async createVectorStore() {
    console.log("Creating document embeddings...");
    this.db = await MemoryVectorStore.fromDocuments(
      this.texts,
      this.selectEmbedding
    );
  }

  /**
   * @description Generate a chunk retriever for the given search type and number of documents.
   */
  createRetriever() {
    console.log("Initialize vector store retriever...");
    this.retriever = this.db.asRetriever({
      k: this.kDocuments,
      searchType: this.searchType,
    });
  }

  async createChain() {
    console.log("Creating Retrieval QA Chain...");

    const prompt = ChatPromptTemplate.fromTemplate(
      `Answer the user's question: {input} based on the following context {context} and chat history:{chat_history}`
    );

    const combineDocsChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
    });

    const chain = await createRetrievalChain({
      combineDocsChain,
      retriever: this.retriever,
    });

    return chain;
  }

  /**
   * @description Returns the chain of the object.
   */
  queryChain() {
    return this.chain;
  }
}

/**
 * RAG Pipeline Optimization Levers
 *
 * Retrieval Augmented Generation (RAG) can be challenging to fine-tune for efficiency. Several key factors influence the performance of the pipeline, including:
 * - **Response speed**: How quickly we can generate answers.
 * - **Answer relevance**: How accurate and contextually appropriate the answers are, and how to minimize hallucinations.
 * - **Answer completeness**: How comprehensive the answers are, ensuring they address the query fully.
 */

const pdfDocument = "./docs/Navpreet_Resume_v2_1.pdf";
// 💡 Count the number of pages in the PDF
const docs = await new PDFLoader(
  path.join(import.meta.dirname, pdfDocument)
).load();
console.log(`Resume Document has ${docs.length} number of pages.`);

/**
 * instantiating our Resume PDF questioner with the following values:
 */
const pdfQa = await new PdfQA({
  model: "llama3",
  pdfDocument,
  chunkSize: 500,
  chunkOverlap: 0,
  searchType: "similarity",
  kDocuments: 3,
}).init();

const pdfQaChain = pdfQa.queryChain();

// we are inputing Question and saving Answer for chat history purpose (in an Array for now)

// QA 1
const answer1 = await pdfQaChain.invoke({
  input: "What is the candidate's name?",
});
console.log("AI => ", answer1.answer, "\n");
const chat_history = [{ Question: answer1.input, Answer: answer1.answer }];

//QA 2
const answer2 = await pdfQaChain.invoke({
  input: "provide me previous question and answer",
  chat_history: chat_history,
});
console.log("AI => ", answer2.answer, "\n");
chat_history.push({ Question: answer2.input, Answer: answer2.answer });

//QA 3
const answer3 = await pdfQaChain.invoke({
  input: "provide me previous question and answer",
  chat_history: chat_history,
});
console.log("AI => ", answer3.answer, "\n");

console.log(chat_history.map((Element) => Element));
