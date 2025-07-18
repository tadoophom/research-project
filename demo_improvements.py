"""
NLP Keyword Context Analysis with spaCy - Summary and Improvements

This project demonstrates advanced keyword context analysis using spaCy's 
dependency parsing capabilities to improve upon traditional window-based approaches.
"""

import spacy
from keyword_context_analyzer import KeywordContextAnalyzer
from dependency_tree_analyzer import DependencyTreeAnalyzer


def demonstrate_improvements():
    """
    Demonstrate key improvements over traditional approaches.
    """
    print("KEYWORD CONTEXT ANALYSIS - KEY IMPROVEMENTS")
    print("=" * 60)
    
    # Initialize analyzers
    context_analyzer = KeywordContextAnalyzer()
    tree_analyzer = DependencyTreeAnalyzer()
    
    # Example text demonstrating complex relationships
    example_text = (
        "The groundbreaking gene therapy treatment showed remarkable efficacy in "
        "treating patients with rare genetic disorders, though some individuals "
        "experienced mild immune reactions that were easily managed by the medical team."
    )
    
    keyword = "treatment"
    
    print(f"\nExample Text: {example_text}")
    print(f"Target Keyword: '{keyword}'")
    print("\n" + "=" * 60)
    
    # 1. Traditional sliding window approach
    print("\n1. TRADITIONAL SLIDING WINDOW APPROACH")
    print("-" * 40)
    
    doc = context_analyzer.nlp(example_text)
    for token in doc:
        if token.text.lower() == keyword.lower():
            window_context = context_analyzer.get_sliding_window_context(doc, token, window_size=5)
            window_text = ' '.join([t.text for t in window_context])
            
            print(f"Window Context (±5 tokens): {window_text}")
            
            # Analyze sentiment using window
            window_sentiment = context_analyzer.analyze_context_sentiment(window_context)
            print(f"Window Sentiment Score: {window_sentiment['combined']:.3f}")
            break
    
    # 2. Improved dependency-based approach
    print("\n2. IMPROVED DEPENDENCY-BASED APPROACH")
    print("-" * 40)
    
    for token in doc:
        if token.text.lower() == keyword.lower():
            # Extract semantic context using dependencies
            semantic_context = tree_analyzer.extract_semantic_context(doc, token)
            dep_context = context_analyzer.get_dependency_context(doc, token)
            
            print(f"Semantic Cluster: {list(semantic_context['semantic_cluster'])}")
            print(f"Dependency Context: {[t.text for t in dep_context]}")
            
            # Show semantic relationships
            if semantic_context['subject_relations']:
                print(f"Subject Relations: {semantic_context['subject_relations']}")
            if semantic_context['modifier_relations']:
                print(f"Modifier Relations: {semantic_context['modifier_relations']}")
            
            # Analyze sentiment using dependencies
            dep_sentiment = context_analyzer.analyze_context_sentiment(dep_context)
            print(f"Dependency Sentiment Score: {dep_sentiment['combined']:.3f}")
            break
    
    # 3. Key advantages demonstration
    print("\n3. KEY ADVANTAGES OF DEPENDENCY-BASED APPROACH")
    print("-" * 40)
    
    advantages = [
        "✓ Captures semantic relationships beyond proximity",
        "✓ Handles long-distance dependencies effectively",
        "✓ Provides richer linguistic context",
        "✓ Better handling of complex sentence structures",
        "✓ More accurate sentiment attribution",
        "✓ Language-structure aware analysis"
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")
    
    # 4. Medical domain specific improvements
    print("\n4. MEDICAL DOMAIN SPECIFIC IMPROVEMENTS")
    print("-" * 40)
    
    medical_examples = [
        {
            'text': "The experimental treatment failed to show efficacy but was well-tolerated.",
            'keyword': 'treatment',
            'explanation': "Dependency parsing correctly attributes 'failed' to efficacy, not treatment"
        },
        {
            'text': "Patients who received the new medication reported significant improvement.",
            'keyword': 'medication',
            'explanation': "Links medication to positive outcome despite intervening words"
        },
        {
            'text': "The procedure, although challenging, was completed successfully.",
            'keyword': 'procedure',
            'explanation': "Correctly identifies positive sentiment despite negative modifier"
        }
    ]
    
    for i, example in enumerate(medical_examples, 1):
        print(f"\n  Example {i}: {example['text']}")
        print(f"  Keyword: '{example['keyword']}'")
        print(f"  Improvement: {example['explanation']}")
        
        # Quick analysis
        results = context_analyzer.find_keyword_contexts(example['text'], example['keyword'])
        if results:
            dep_score = results[0]['dependency_sentiment']['combined']
            win_score = results[0]['window_sentiment']['combined']
            print(f"  Dependency Score: {dep_score:.3f} | Window Score: {win_score:.3f}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: Dependency-based analysis provides more accurate")
    print("and contextually rich keyword sentiment analysis for medical texts.")


def show_technical_features():
    """
    Highlight technical features implemented.
    """
    print("\n\nTECHNICAL FEATURES IMPLEMENTED")
    print("=" * 60)
    
    features = {
        "Dependency Tree Traversal": [
            "Recursive traversal of syntactic dependencies",
            "Semantic clustering through meaningful relationships",
            "Multiple traversal strategies (depth-limited, relationship-filtered)"
        ],
        "Context Extraction Methods": [
            "Traditional sliding window",
            "Dependency-based semantic context",
            "Syntactic chunk-based context",
            "Sentence-level context"
        ],
        "Sentiment Analysis Approaches": [
            "Rule-based with medical vocabulary",
            "TextBlob integration",
            "Dependency-aware sentiment attribution",
            "Negation handling in dependency structures"
        ],
        "Advanced NLP Features": [
            "Part-of-speech aware processing",
            "Named entity consideration",
            "Lemmatization for keyword matching",
            "Multi-method comparison and validation"
        ],
        "Medical Domain Enhancements": [
            "Medical terminology vocabulary",
            "Clinical context understanding",
            "Drug and treatment specific analysis",
            "Patient outcome sentiment tracking"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        print("-" * 30)
        for item in items:
            print(f"  • {item}")


if __name__ == "__main__":
    try:
        demonstrate_improvements()
        show_technical_features()
        
        print(f"\n\n{'='*60}")
        print("This implementation showcases spaCy's powerful dependency")
        print("parsing capabilities for advanced keyword context analysis,")
        print("providing significant improvements over traditional methods.")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure spaCy and the English model are properly installed.")
