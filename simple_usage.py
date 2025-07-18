#!/usr/bin/env python3
"""
Simple usage example for the NLP Keyword Context Analyzer

This script demonstrates how to use the analyzer with your own texts and keywords.
"""

from keyword_context_analyzer import KeywordContextAnalyzer
from dependency_tree_analyzer import DependencyTreeAnalyzer


def analyze_text_examples():
    """
    Simple example showing how to analyze custom text.
    """
    # Initialize the analyzer
    analyzer = KeywordContextAnalyzer()
    
    # Your custom texts
    texts = [
        "The new treatment protocol showed excellent results in clinical trials.",
        "Patients responded poorly to the experimental drug therapy.",
        "The surgical procedure was completed successfully with minimal complications.",
        "Medical researchers developed an innovative diagnostic technique.",
        "The therapy failed to meet the primary efficacy endpoints."
    ]
    
    # Keywords to analyze
    keywords = ["treatment", "drug", "procedure"]
    
    print("SIMPLE KEYWORD CONTEXT ANALYSIS")
    print("=" * 50)
    
    for keyword in keywords:
        print(f"\nAnalyzing keyword: '{keyword}'")
        print("-" * 30)
        
        # Analyze the texts
        results_df = analyzer.analyze_text_collection(texts, keyword)
        
        if not results_df.empty:
            # Print results for each occurrence
            for idx, row in results_df.iterrows():
                sentiment_class = analyzer.classify_sentiment(row['dep_combined'])
                
                print(f"\nText: {row['sentence']}")
                print(f"Context: {row['dependency_context']}")
                print(f"Sentiment: {row['dep_combined']:.3f} ({sentiment_class})")
        else:
            print(f"No occurrences of '{keyword}' found.")


def quick_single_text_analysis():
    """
    Quick analysis of a single text.
    """
    print(f"\n\n{'='*50}")
    print("QUICK SINGLE TEXT ANALYSIS")
    print("=" * 50)
    
    analyzer = KeywordContextAnalyzer()
    
    # Single text example
    text = "The revolutionary cancer treatment demonstrated outstanding efficacy in reducing tumor size, though some patients experienced manageable side effects."
    keyword = "treatment"
    
    print(f"Text: {text}")
    print(f"Keyword: '{keyword}'")
    print("-" * 50)
    
    # Find keyword contexts
    contexts = analyzer.find_keyword_contexts(text, keyword)
    
    for context in contexts:
        print(f"\nDependency Context: {context['dependency_context']}")
        print(f"Window Context: {context['window_context']}")
        
        dep_sentiment = context['dependency_sentiment']['combined']
        win_sentiment = context['window_sentiment']['combined']
        
        print(f"Dependency Sentiment: {dep_sentiment:.3f} ({analyzer.classify_sentiment(dep_sentiment)})")
        print(f"Window Sentiment: {win_sentiment:.3f} ({analyzer.classify_sentiment(win_sentiment)})")
        
        # Show dependency relationships
        print(f"Dependency Relations: {context['dependency_relations']}")


def show_dependency_tree():
    """
    Show dependency tree structure for educational purposes.
    """
    print(f"\n\n{'='*50}")
    print("DEPENDENCY TREE VISUALIZATION")
    print("=" * 50)
    
    tree_analyzer = DependencyTreeAnalyzer()
    
    text = "The experimental drug showed promising results in treating patients."
    keyword = "drug"
    
    print(f"Text: {text}")
    print(f"Keyword: '{keyword}'")
    print("-" * 50)
    
    # Visualize dependency tree
    tree_analyzer.visualize_dependency_tree(text, keyword)


if __name__ == "__main__":
    try:
        # Run all examples
        analyze_text_examples()
        quick_single_text_analysis()
        show_dependency_tree()
        
        print(f"\n\n{'='*50}")
        print("USAGE TIPS:")
        print("=" * 50)
        print("1. Initialize KeywordContextAnalyzer() for basic analysis")
        print("2. Use analyze_text_collection() for multiple texts")
        print("3. Use find_keyword_contexts() for single text analysis")
        print("4. Use DependencyTreeAnalyzer() for detailed linguistic analysis")
        print("5. Check analysis_report.txt and detailed_results.json for full results")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed: uv sync")
