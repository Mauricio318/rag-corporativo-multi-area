# 🏢 RAG Corporativo Multi-Área

Um sistema de Inteligência Artificial corporativo completo construído com **Python, Streamlit, LangChain, ChromaDB e Google Gemini 2.5 Flash**. Este projeto permite que usuários façam upload de documentos em PDF/Word/TXT, indexem o conhecimento de forma segmentada por áreas da empresa (RH, Suporte, Financeiro, etc.) e conversem com os documentos através de um chatbot inteligente que preserva a segurança e a veracidade da informação (retrieval-augmented generation).

## 🚀 Funcionalidades Principais

- **Múltiplos Perfis de Acesso**: Suporte nativo para perfil "Usuário" e "Admin".
- **Painel Administrativo Moderno**: Upload rigoroso de arquivos, gestão dinâmica da base de dados vetorial por áreas e auditoria de logs.
- **Gestão Segura de Credenciais**: Interface inteligente para gerenciar chaves da API do Google Cloud localmente (via Secrets).
- **RAG Estrito e Confiável**: O LLM está travado com "System Prompts" severos para **nunca** alucinar respostas fora dos documentos carregados. 
- **Memória de Curto Prazo (Conversacional)**: O bot se recorda de turnos anteriores do chat sem misturar o contexto de áreas distintas (isolamento de memória stateful).
- **Busca Híbrida**: Capacidade de procurar conhecimento apenas na área selecionada ou no modo de busca Global.

## 🛠 Tecnologias Utilizadas

- **Frontend**: Streamlit + Extra Streamlit Components
- **Backend & Orquestração**: Python 3.x + LangChain
- **Banco de Dados Vetorial**: ChromaDB (Local/In-Memory persistente)
- **Motor de Embeddings**: HuggingFace SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM Principal**: Google Generative AI (`gemini-2.5-flash`)
- **Processamento de Arquivos**: PyMuPDF (`pypdf`), python-docx

## 📥 Como Rodar o Projeto (Localmente)

1. Clone este repositório.
2. Instale as dependências:
   ```bash
   pip install streamlit langchain langchain-google-genai langchain-chroma sentence-transformers pypdf python-docx extra-streamlit-components
   ```
3. Execute a aplicação (ela gerará as pastas e configs necessárias automaticamente):
   ```bash
   python -m streamlit run app_rag_chroma_v2.py
   ```
4. Navegue até as **Opções Avançadas** na barra lateral, entre como Admin e insira a sua credencial do Google AI Studio para ativar o motor LLM.

## 🛡 Considerações de Segurança

O arquivo `.streamlit/secrets.toml`, que armazena a sua chave de API, **jamais** deve ser feito commit no GitHub. O repositório já conta com um `.gitignore` restrito blindando todos os logs e bases de dados do Chroma (`chroma_db/`).
