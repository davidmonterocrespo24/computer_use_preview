"""
Example scripts for using the browser agent.
"""

# Example 1: Simple search
EXAMPLE_1 = {
    "name": "Simple Google Search",
    "query": "Search Google for 'Python web scraping tutorial'",
    "expected_actions": [
        "Find search input field",
        "Type search query",
        "Press Enter or click search button",
        "Wait for results",
        "Return search results"
    ]
}

# Example 2: Multi-step navigation
EXAMPLE_2 = {
    "name": "Wikipedia Search",
    "query": "Go to Wikipedia and search for 'Machine Learning', then summarize the first paragraph",
    "expected_actions": [
        "Navigate to wikipedia.org",
        "Find search box",
        "Search for 'Machine Learning'",
        "Read first paragraph",
        "Return summary"
    ]
}

# Example 3: Form filling
EXAMPLE_3 = {
    "name": "Form Interaction",
    "query": "Find a contact form and fill in name 'John Doe', email 'john@example.com'",
    "expected_actions": [
        "Locate form fields",
        "Fill name field",
        "Fill email field",
        "Confirm completion"
    ]
}

# Example 4: E-commerce interaction
EXAMPLE_4 = {
    "name": "Product Search",
    "query": "Go to Amazon and search for 'laptop', then tell me the price of the first result",
    "expected_actions": [
        "Navigate to amazon.com",
        "Find search box",
        "Search for 'laptop'",
        "Find first product",
        "Extract price",
        "Return price information"
    ]
}

# Example 5: News extraction
EXAMPLE_5 = {
    "name": "News Headlines",
    "query": "Go to BBC News and get the top 3 headlines",
    "expected_actions": [
        "Navigate to bbc.com/news",
        "Identify headline elements",
        "Extract top 3 headlines",
        "Return headlines"
    ]
}


def run_example(example_num: int = 1):
    """
    Run an example query.
    
    Usage:
        python -c "from examples import run_example; run_example(1)"
    """
    import subprocess
    
    examples = [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5]
    
    if example_num < 1 or example_num > len(examples):
        print(f"Invalid example number. Choose 1-{len(examples)}")
        return
    
    example = examples[example_num - 1]
    
    print(f"\n{'='*60}")
    print(f"Running Example {example_num}: {example['name']}")
    print(f"{'='*60}")
    print(f"\nQuery: {example['query']}")
    print(f"\nExpected Actions:")
    for i, action in enumerate(example['expected_actions'], 1):
        print(f"  {i}. {action}")
    print(f"\n{'='*60}\n")
    
    # Run the main script
    cmd = [
        "python", "main.py",
        "--query", example['query'],
        "--save-conversation", f"example_{example_num}_conversation.json"
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_example(int(sys.argv[1]))
    else:
        print("Available Examples:")
        print("-" * 60)
        for i, ex in enumerate([EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5], 1):
            print(f"{i}. {ex['name']}")
            print(f"   Query: {ex['query']}")
            print()
        
        print("\nUsage: python examples.py <example_number>")
        print("Example: python examples.py 1")
