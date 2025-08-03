# AI Agents Collection 🤖

A comprehensive collection of AI-powered agents for various automation tasks, including cover letter generation, fraud detection, and more.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- ChromaDB server (for vector storage)
- HuggingFace API token

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anbarasans-arch/agents.git
   cd agents
   ```

2. **Install dependencies:**
   ```bash
   # Install all dependencies automatically
   pip install -e .
   
   # Or install in development mode with optional dependencies
   pip install -e ".[dev,jupyter]"
   ```

3. **Set up environment variables:**
   ```bash
   # Create a .env file in the root directory
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

4. **Start ChromaDB server:**
   ```bash
   # Using Docker (recommended)
   docker run -p 8000:8000 chromadb/chroma
   
   # Or install and run locally
   pip install chromadb
   chroma run --host localhost --port 8000
   ```

## 🎯 Available Agents

### 1. Cover Letter Generator Agent

Generate tailored cover letters that match your resume to specific job descriptions.

**Features:**
- 📄 Multi-format resume support (PDF, DOCX, DOC)
- 🔗 Automatic job description extraction from URLs
- 🧠 AI-powered content matching
- 🌐 Modern web UI with Streamlit
- 💾 Vector-based resume storage

**Usage:**

**Web UI (Recommended):**
```bash
streamlit run myagents/tailored_coverletter_agent_ui.py
```

**Command Line:**
```bash
python myagents/tailored_coverletter_agent.py
```

**Using pip scripts:**
```bash
# After installation with pip install -e .
cover-letter-ui    # Launch web interface
cover-letter-agent # Launch command-line version
```

### 2. Fraud Detection Agent

AI-powered fraud detection system for financial transactions.

```bash
python myagents/CCfrauddetectionagent.py
```

## 📁 Project Structure

```
agents/
├── myagents/                    # Main agents directory
│   ├── tailored_coverletter_agent.py      # CLI cover letter agent
│   ├── tailored_coverletter_agent_ui.py   # Web UI cover letter agent
│   ├── CCfrauddetectionagent.py           # Fraud detection agent
│   └── ...                                # Other agents
├── config/                      # Configuration files
├── tools/                       # Utility tools
├── utils/                       # Helper utilities
├── logger/                      # Logging configuration
├── exception/                   # Custom exceptions
├── pyproject.toml              # Project dependencies
├── requirements.txt            # Legacy requirements
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required
HF_TOKEN=your_huggingface_token_here

# Optional
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### ChromaDB Setup

The vector database is required for the cover letter agent:

**Option 1: Docker (Recommended)**
```bash
docker run -p 8000:8000 chromadb/chroma
```

**Option 2: Local Installation**
```bash
pip install chromadb
chroma run --host localhost --port 8000
```

## 📦 Dependencies

Core dependencies are automatically installed via `pyproject.toml`:

- **AI/ML**: LangChain, HuggingFace Transformers, ChromaDB
- **Document Processing**: PyPDF, python-docx, docx2txt
- **Web Scraping**: BeautifulSoup4, requests
- **Web UI**: Streamlit
- **Utilities**: python-dotenv, pandas, numpy

## 🎨 Features by Agent

### Cover Letter Generator
- ✅ Multi-format resume upload (PDF, DOCX, DOC)
- ✅ URL-based job description extraction
- ✅ AI-powered content matching
- ✅ Vector similarity search
- ✅ Modern web interface
- ✅ Download generated letters
- ✅ Session state management

### Fraud Detection
- ✅ Machine learning-based detection
- ✅ Transaction pattern analysis
- ✅ Real-time scoring
- ✅ Configurable thresholds

## 🚀 Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/anbarasans-arch/agents.git
cd agents
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black myagents/
isort myagents/

# Lint code
flake8 myagents/
```

### Adding New Agents

1. Create your agent in the `myagents/` directory
2. Add any new dependencies to `pyproject.toml`
3. Update this README with usage instructions
4. Add tests if applicable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**ChromaDB Connection Failed:**
```bash
# Ensure ChromaDB server is running
docker run -p 8000:8000 chromadb/chroma
```

**HuggingFace Token Error:**
```bash
# Set your token in .env file
echo "HF_TOKEN=your_token_here" > .env
```

**Document Processing Error:**
```bash
# Install additional dependencies
pip install python-docx docx2txt lxml
```

**Streamlit Import Error:**
```bash
# Reinstall streamlit
pip install --upgrade streamlit
```

### Getting Help

- 📝 Check the [Issues](https://github.com/anbarasans-arch/agents/issues) page
- 💬 Start a [Discussion](https://github.com/anbarasans-arch/agents/discussions)
- 📧 Contact: [anbarasans.arch@gmail.com](mailto:anbarasans.arch@gmail.com)

## 🌟 Acknowledgments

- HuggingFace for transformer models
- LangChain for AI framework
- ChromaDB for vector storage
- Streamlit for web interface
- Open source community for inspiration

---

**Made with ❤️ by [Anbarasan](https://github.com/anbarasans-arch)**
