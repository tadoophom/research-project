import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
from collections import defaultdict


class KeywordContextAnalyzer:
    """
    Analyzes keywords within text to determine if they appear in positive or negative contexts.
    Uses spaCy's dependency parsing and NLP capabilities for context window analysis.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the analyzer with a spaCy model."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install it using:")
            print(f"python -m spacy download {model_name}")
            raise
        
        # Positive and negative sentiment indicators
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'positive', 'effective', 'beneficial', 'helpful', 'successful', 
            'improved', 'better', 'best', 'superior', 'optimal', 'enhanced',
            'promising', 'favorable', 'advantageous', 'valuable', 'significant'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'negative',
            'harmful', 'dangerous', 'ineffective', 'failed', 'worse', 'worst',
            'inferior', 'problematic', 'concerning', 'adverse', 'detrimental',
            'unfavorable', 'disappointing', 'insufficient', 'inadequate'
        }
    
    def get_dependency_context(self, doc: spacy.tokens.Doc, keyword_token: spacy.tokens.Token, 
                             max_depth: int = 3) -> List[spacy.tokens.Token]:
        """
        Extract context tokens using dependency tree traversal.
        
        Args:
            doc: spaCy document
            keyword_token: The target keyword token
            max_depth: Maximum depth to traverse in dependency tree
            
        Returns:
            List of context tokens
        """
        context_tokens = set([keyword_token])
        
        def traverse_dependencies(token: spacy.tokens.Token, depth: int):
            if depth > max_depth:
                return
            
            # Add children
            for child in token.children:
                if child not in context_tokens:
                    context_tokens.add(child)
                    traverse_dependencies(child, depth + 1)
            
            # Add head (parent)
            if token.head != token and token.head not in context_tokens:
                context_tokens.add(token.head)
                if depth < max_depth:
                    traverse_dependencies(token.head, depth + 1)
        
        traverse_dependencies(keyword_token, 0)
        return list(context_tokens)
    
    def get_sliding_window_context(self, doc: spacy.tokens.Doc, keyword_token: spacy.tokens.Token, 
                                 window_size: int = 5) -> List[spacy.tokens.Token]:
        """
        Extract context using a sliding window approach.
        
        Args:
            doc: spaCy document
            keyword_token: The target keyword token
            window_size: Number of tokens on each side of the keyword
            
        Returns:
            List of context tokens
        """
        keyword_idx = keyword_token.i
        start_idx = max(0, keyword_idx - window_size)
        end_idx = min(len(doc), keyword_idx + window_size + 1)
        
        return [token for token in doc[start_idx:end_idx]]
    
    def analyze_sentiment_with_textblob(self, text: str) -> float:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_context_sentiment(self, context_tokens: List[spacy.tokens.Token]) -> Dict[str, float]:
        """
        Analyze sentiment of context tokens using multiple approaches.
        
        Args:
            context_tokens: List of tokens in the context
            
        Returns:
            Dictionary with sentiment scores from different methods
        """
        # Method 1: Rule-based using predefined word lists
        positive_count = 0
        negative_count = 0
        total_sentiment_words = 0
        
        for token in context_tokens:
            lemma = token.lemma_.lower()
            if lemma in self.positive_words:
                positive_count += 1
                total_sentiment_words += 1
            elif lemma in self.negative_words:
                negative_count += 1
                total_sentiment_words += 1
        
        rule_based_score = 0.0
        if total_sentiment_words > 0:
            rule_based_score = (positive_count - negative_count) / total_sentiment_words
        
        # Method 2: TextBlob sentiment analysis
        context_text = ' '.join([token.text for token in context_tokens])
        textblob_score = self.analyze_sentiment_with_textblob(context_text)
        
        # Method 3: Consider dependency relationships
        dependency_score = self._analyze_dependency_sentiment(context_tokens)
        
        return {
            'rule_based': rule_based_score,
            'textblob': textblob_score,
            'dependency_based': dependency_score,
            'combined': (rule_based_score + textblob_score + dependency_score) / 3
        }
    
    def _analyze_dependency_sentiment(self, context_tokens: List[spacy.tokens.Token]) -> float:
        """Analyze sentiment considering dependency relationships."""
        sentiment_score = 0.0
        sentiment_count = 0
        
        for token in context_tokens:
            # Check for negation dependencies
            is_negated = any(child.dep_ == 'neg' for child in token.children)
            
            lemma = token.lemma_.lower()
            token_sentiment = 0.0
            
            if lemma in self.positive_words:
                token_sentiment = 1.0
            elif lemma in self.negative_words:
                token_sentiment = -1.0
            
            if token_sentiment != 0.0:
                # Reverse sentiment if negated
                if is_negated:
                    token_sentiment *= -1
                
                sentiment_score += token_sentiment
                sentiment_count += 1
        
        return sentiment_score / sentiment_count if sentiment_count > 0 else 0.0
    
    def find_keyword_contexts(self, text: str, keyword: str) -> List[Dict]:
        """
        Find all occurrences of a keyword and analyze their contexts.
        
        Args:
            text: Input text to analyze
            keyword: Keyword to search for
            
        Returns:
            List of dictionaries containing context analysis results
        """
        doc = self.nlp(text)
        keyword_lower = keyword.lower()
        results = []
        
        for token in doc:
            if token.text.lower() == keyword_lower or token.lemma_.lower() == keyword_lower:
                # Get context using both methods
                dep_context = self.get_dependency_context(doc, token)
                window_context = self.get_sliding_window_context(doc, token)
                
                # Analyze sentiment for both context types
                dep_sentiment = self.analyze_context_sentiment(dep_context)
                window_sentiment = self.analyze_context_sentiment(window_context)
                
                result = {
                    'keyword_position': token.i,
                    'sentence': token.sent.text.strip(),
                    'dependency_context': [t.text for t in dep_context],
                    'window_context': [t.text for t in window_context],
                    'dependency_sentiment': dep_sentiment,
                    'window_sentiment': window_sentiment,
                    'dependency_relations': [(t.text, t.dep_, t.head.text) for t in dep_context],
                }
                
                results.append(result)
        
        return results
    
    def analyze_text_collection(self, texts: List[str], keyword: str) -> pd.DataFrame:
        """
        Analyze a collection of texts for keyword sentiment contexts.
        
        Args:
            texts: List of texts to analyze
            keyword: Keyword to search for
            
        Returns:
            DataFrame with analysis results
        """
        all_results = []
        
        for text_idx, text in enumerate(texts):
            contexts = self.find_keyword_contexts(text, keyword)
            
            for context in contexts:
                result_row = {
                    'text_id': text_idx,
                    'keyword': keyword,
                    'sentence': context['sentence'],
                    'keyword_position': context['keyword_position'],
                    'dep_rule_based': context['dependency_sentiment']['rule_based'],
                    'dep_textblob': context['dependency_sentiment']['textblob'],
                    'dep_dependency_based': context['dependency_sentiment']['dependency_based'],
                    'dep_combined': context['dependency_sentiment']['combined'],
                    'win_rule_based': context['window_sentiment']['rule_based'],
                    'win_textblob': context['window_sentiment']['textblob'],
                    'win_dependency_based': context['window_sentiment']['dependency_based'],
                    'win_combined': context['window_sentiment']['combined'],
                    'dependency_context': ', '.join(context['dependency_context']),
                    'window_context': ', '.join(context['window_context']),
                }
                all_results.append(result_row)
        
        return pd.DataFrame(all_results)
    
    def classify_sentiment(self, sentiment_score: float, threshold: float = 0.1) -> str:
        """Classify sentiment score into positive, negative, or neutral."""
        if sentiment_score > threshold:
            return 'positive'
        elif sentiment_score < -threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate a summary report of the analysis."""
        if df.empty:
            return {'error': 'No data to analyze'}
        
        summary = {
            'total_occurrences': len(df),
            'dependency_analysis': {
                'positive': len(df[df['dep_combined'] > 0.1]),
                'negative': len(df[df['dep_combined'] < -0.1]),
                'neutral': len(df[abs(df['dep_combined']) <= 0.1]),
                'avg_sentiment': df['dep_combined'].mean(),
            },
            'window_analysis': {
                'positive': len(df[df['win_combined'] > 0.1]),
                'negative': len(df[df['win_combined'] < -0.1]),
                'neutral': len(df[abs(df['win_combined']) <= 0.1]),
                'avg_sentiment': df['win_combined'].mean(),
            }
        }
        
        return summary


def demonstrate_analysis():
    """Demonstrate the keyword context analysis with medical examples."""
    analyzer = KeywordContextAnalyzer()
    
    # Medical text examples
    medical_texts = [
        "The new treatment showed excellent results in reducing pain levels. Patients reported significant improvement in their quality of life.",
        "Unfortunately, the medication caused severe side effects including nausea and dizziness. The treatment was discontinued due to adverse reactions.",
        "The diagnostic procedure was successful and helped identify the underlying condition. Early detection is crucial for effective treatment.",
        "The patient experienced complications during surgery, but the medical team handled the situation professionally. Recovery was slower than expected.",
        "Clinical trials demonstrated that the drug is highly effective against the target disease. The treatment shows promising results.",
        "The therapy failed to provide adequate relief for chronic pain patients. Alternative approaches need to be considered.",
        "Regular exercise and proper medication management led to excellent patient outcomes. The holistic approach was beneficial.",
        "The new imaging technique revealed concerning abnormalities that require immediate attention and further investigation."
    ]
    
    # Analyze for different keywords
    keywords = ['treatment', 'medication', 'patient']
    
    for keyword in keywords:
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR KEYWORD: '{keyword.upper()}'")
        print(f"{'='*60}")
        
        # Perform analysis
        results_df = analyzer.analyze_text_collection(medical_texts, keyword)
        
        if not results_df.empty:
            # Generate summary
            summary = analyzer.generate_summary_report(results_df)
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"Total occurrences: {summary['total_occurrences']}")
            
            print(f"\nDependency-based Analysis:")
            dep_analysis = summary['dependency_analysis']
            print(f"  Positive contexts: {dep_analysis['positive']}")
            print(f"  Negative contexts: {dep_analysis['negative']}")
            print(f"  Neutral contexts: {dep_analysis['neutral']}")
            print(f"  Average sentiment: {dep_analysis['avg_sentiment']:.3f}")
            
            print(f"\nWindow-based Analysis:")
            win_analysis = summary['window_analysis']
            print(f"  Positive contexts: {win_analysis['positive']}")
            print(f"  Negative contexts: {win_analysis['negative']}")
            print(f"  Neutral contexts: {win_analysis['neutral']}")
            print(f"  Average sentiment: {win_analysis['avg_sentiment']:.3f}")
            
            print(f"\nDETAILED RESULTS:")
            print("-" * 60)
            
            for idx, row in results_df.iterrows():
                dep_class = analyzer.classify_sentiment(row['dep_combined'])
                win_class = analyzer.classify_sentiment(row['win_combined'])
                
                print(f"\nText {row['text_id'] + 1}: {row['sentence']}")
                print(f"Dependency context: {row['dependency_context']}")
                print(f"Window context: {row['window_context']}")
                print(f"Dependency sentiment: {row['dep_combined']:.3f} ({dep_class})")
                print(f"Window sentiment: {row['win_combined']:.3f} ({win_class})")
        else:
            print(f"No occurrences of '{keyword}' found in the texts.")


if __name__ == "__main__":
    demonstrate_analysis()
