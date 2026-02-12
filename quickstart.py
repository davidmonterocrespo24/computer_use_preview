"""
Quick start script - Interactive CLI for the browser agent.
"""
import sys
import os


def print_header():
    print("""
╔═══════════════════════════════════════════════════════════╗
║           🌐 Browser Agent - Quick Start 🌐              ║
╚═══════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """Check if environment is properly configured."""
    issues = []
    
    # Check for .env file
    if not os.path.exists('.env'):
        issues.append("❌ .env file not found. Copy .env.example to .env")
    
    # Check for API keys
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        if os.path.exists('.env'):
            issues.append("⚠️  No API keys found in .env file")
    
    # Check Playwright installation
    try:
        import playwright
    except ImportError:
        issues.append("❌ Playwright not installed. Run: pip install playwright")
    
    return issues


def select_provider():
    """Select LLM provider."""
    print("\n📡 Select LLM Provider:")
    print("  1. OpenAI (GPT-4o-mini, GPT-4)")
    print("  2. Anthropic (Claude)")
    print("  3. Local Model (Ollama, llama.cpp, etc.)")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == "1":
            return "openai", "gpt-4o-mini"
        elif choice == "2":
            return "anthropic", "claude-3-5-sonnet-20241022"
        elif choice == "3":
            model = input("Enter model name (e.g., phi3): ").strip() or "phi3"
            return "local", model
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def select_task():
    """Select or enter a task."""
    print("\n📝 Select a task or enter your own:")
    print("  1. Search Google for information")
    print("  2. Navigate to Wikipedia and search")
    print("  3. Get news headlines")
    print("  4. Custom query")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            topic = input("What do you want to search for? ").strip()
            return f"Search Google for '{topic}' and tell me the key findings"
        
        elif choice == "2":
            topic = input("What Wikipedia article? ").strip()
            return f"Go to Wikipedia, search for '{topic}', and summarize the introduction"
        
        elif choice == "3":
            source = input("News source (e.g., BBC, CNN): ").strip() or "BBC"
            return f"Go to {source} news and get the top 3 headlines"
        
        elif choice == "4":
            return input("Enter your query: ").strip()
        
        else:
            print("Invalid choice. Please enter 1-4.")


def main():
    print_header()
    
    # Check environment
    issues = check_environment()
    if issues:
        print("⚠️  Environment Issues:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease fix these issues first.")
        
        if input("\nContinue anyway? (y/n): ").lower() != 'y':
            return 1
    
    # Select provider
    provider, model = select_provider()
    
    # Select task
    query = select_task()
    
    # Ask for options
    print("\n⚙️  Options:")
    headless = input("Run in headless mode? (y/n): ").lower() == 'y'
    save_conversation = input("Save conversation? (y/n): ").lower() == 'y'
    
    # Build command
    cmd = [
        "python", "main.py",
        "--query", query,
        "--provider", provider,
        "--model", model
    ]
    
    if headless:
        cmd.append("--headless")
    
    if save_conversation:
        filename = f"conversation_{provider}_{model.replace('/', '_')}.json"
        cmd.extend(["--save-conversation", filename])
    
    # Show summary
    print("\n" + "="*60)
    print("🚀 Running Browser Agent")
    print("="*60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Query: {query}")
    print(f"Headless: {headless}")
    print("="*60 + "\n")
    
    # Run
    import subprocess
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
