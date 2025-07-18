import spacy
from typing import List, Dict, Tuple, Set
import json


class DependencyTreeAnalyzer:
    """
    Advanced dependency tree analysis for keyword context detection.
    Demonstrates spaCy's built-in context tree structure generation.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize with spaCy model."""
        self.nlp = spacy.load(model_name)
    
    def visualize_dependency_tree(self, text: str, keyword: str) -> None:
        """
        Display dependency tree structure for sentences containing the keyword.
        """
        doc = self.nlp(text)
        keyword_lower = keyword.lower()
        
        for sent in doc.sents:
            keyword_found = any(token.text.lower() == keyword_lower or 
                              token.lemma_.lower() == keyword_lower 
                              for token in sent)
            
            if keyword_found:
                print(f"\nSentence: {sent.text}")
                print("Dependency Tree Structure:")
                print("-" * 50)
                
                for token in sent:
                    # Mark keyword tokens
                    keyword_marker = " <-- KEYWORD" if (token.text.lower() == keyword_lower or 
                                                       token.lemma_.lower() == keyword_lower) else ""
                    
                    print(f"{token.text:15} | {token.pos_:10} | {token.dep_:15} | "
                          f"Head: {token.head.text:15} | Children: {[child.text for child in token.children]}{keyword_marker}")
    
    def extract_semantic_context(self, doc: spacy.tokens.Doc, keyword_token: spacy.tokens.Token) -> Dict:
        """
        Extract semantic context using dependency relationships.
        Focuses on semantically meaningful connections rather than just proximity.
        """
        context = {
            'subject_relations': [],
            'object_relations': [],
            'modifier_relations': [],
            'complement_relations': [],
            'coordination_relations': [],
            'semantic_cluster': set()
        }
        
        # Find direct semantic relationships
        self._find_subject_object_relations(keyword_token, context)
        self._find_modifier_relations(keyword_token, context)
        self._find_complement_relations(keyword_token, context)
        self._find_coordination_relations(keyword_token, context)
        
        # Build semantic cluster through dependency traversal
        self._build_semantic_cluster(keyword_token, context['semantic_cluster'])
        
        return context
    
    def _find_subject_object_relations(self, token: spacy.tokens.Token, context: Dict) -> None:
        """Find subject and object relationships."""
        # If token is a verb, find its subjects and objects
        if token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass', 'csubj']:
                    context['subject_relations'].append({
                        'relation': child.dep_,
                        'subject': child.text,
                        'verb': token.text
                    })
                elif child.dep_ in ['dobj', 'iobj', 'pobj']:
                    context['object_relations'].append({
                        'relation': child.dep_,
                        'object': child.text,
                        'verb': token.text
                    })
        
        # If token is a subject/object, find the verb it relates to
        if token.dep_ in ['nsubj', 'nsubjpass', 'csubj']:
            context['subject_relations'].append({
                'relation': token.dep_,
                'subject': token.text,
                'verb': token.head.text
            })
        elif token.dep_ in ['dobj', 'iobj', 'pobj']:
            context['object_relations'].append({
                'relation': token.dep_,
                'object': token.text,
                'verb': token.head.text
            })
    
    def _find_modifier_relations(self, token: spacy.tokens.Token, context: Dict) -> None:
        """Find modifier relationships (adjectives, adverbs, etc.)."""
        # Find modifiers of the token
        for child in token.children:
            if child.dep_ in ['amod', 'advmod', 'compound', 'npadvmod']:
                context['modifier_relations'].append({
                    'modifier': child.text,
                    'modified': token.text,
                    'relation': child.dep_
                })
        
        # If token is a modifier, find what it modifies
        if token.dep_ in ['amod', 'advmod', 'compound', 'npadvmod']:
            context['modifier_relations'].append({
                'modifier': token.text,
                'modified': token.head.text,
                'relation': token.dep_
            })
    
    def _find_complement_relations(self, token: spacy.tokens.Token, context: Dict) -> None:
        """Find complement relationships."""
        for child in token.children:
            if child.dep_ in ['xcomp', 'ccomp', 'acomp']:
                context['complement_relations'].append({
                    'complement': child.text,
                    'head': token.text,
                    'relation': child.dep_
                })
    
    def _find_coordination_relations(self, token: spacy.tokens.Token, context: Dict) -> None:
        """Find coordination (and, or) relationships."""
        # Find coordinated elements
        for child in token.children:
            if child.dep_ == 'conj':
                context['coordination_relations'].append({
                    'coordinated_with': child.text,
                    'original': token.text,
                    'coordinator': self._find_coordinator(token, child)
                })
    
    def _find_coordinator(self, token1: spacy.tokens.Token, token2: spacy.tokens.Token) -> str:
        """Find the coordinating conjunction between two tokens."""
        for child in token1.children:
            if child.dep_ == 'cc' and child.i < token2.i:
                return child.text
        return 'unknown'
    
    def _build_semantic_cluster(self, token: spacy.tokens.Token, cluster: Set, 
                               max_depth: int = 2, current_depth: int = 0) -> None:
        """Build a semantic cluster through meaningful dependency traversal."""
        if current_depth > max_depth or token in cluster:
            return
        
        cluster.add(token.text)
        
        # Traverse meaningful relationships only
        meaningful_deps = {
            'nsubj', 'nsubjpass', 'dobj', 'iobj', 'amod', 'advmod', 
            'compound', 'conj', 'xcomp', 'ccomp', 'pobj', 'prep'
        }
        
        for child in token.children:
            if child.dep_ in meaningful_deps:
                self._build_semantic_cluster(child, cluster, max_depth, current_depth + 1)
        
        if token.head != token and token.dep_ in meaningful_deps:
            self._build_semantic_cluster(token.head, cluster, max_depth, current_depth + 1)
    
    def compare_context_methods(self, text: str, keyword: str, window_size: int = 5) -> Dict:
        """
        Compare different context extraction methods.
        """
        doc = self.nlp(text)
        keyword_lower = keyword.lower()
        comparison_results = []
        
        for token in doc:
            if token.text.lower() == keyword_lower or token.lemma_.lower() == keyword_lower:
                # Method 1: Traditional sliding window
                window_start = max(0, token.i - window_size)
                window_end = min(len(doc), token.i + window_size + 1)
                window_context = [t.text for t in doc[window_start:window_end]]
                
                # Method 2: Dependency-based context
                semantic_context = self.extract_semantic_context(doc, token)
                
                # Method 3: Sentence-level context
                sentence_context = [t.text for t in token.sent]
                
                # Method 4: Syntactic chunk context
                chunk_context = self._get_chunk_context(token)
                
                result = {
                    'keyword_position': token.i,
                    'sentence': token.sent.text.strip(),
                    'window_context': window_context,
                    'semantic_context': semantic_context,
                    'sentence_context': sentence_context,
                    'chunk_context': chunk_context,
                    'context_sizes': {
                        'window': len(window_context),
                        'semantic_cluster': len(semantic_context['semantic_cluster']),
                        'sentence': len(sentence_context),
                        'chunk': len(chunk_context)
                    }
                }
                
                comparison_results.append(result)
        
        return comparison_results
    
    def _get_chunk_context(self, token: spacy.tokens.Token) -> List[str]:
        """Get context based on syntactic chunks (noun phrases, etc.)."""
        context_tokens = set()
        
        # Add tokens from the same noun chunk
        if token.dep_ != 'ROOT':
            for chunk in token.doc.noun_chunks:
                if token in chunk:
                    context_tokens.update(chunk)
        
        # Add related verb phrases
        if token.pos_ in ['NOUN', 'PROPN']:
            # Find governing verb
            head = token.head
            while head.pos_ != 'VERB' and head.dep_ != 'ROOT' and head != head.head:
                head = head.head
            if head.pos_ == 'VERB':
                # Add verb and its direct dependents
                context_tokens.add(head)
                context_tokens.update(head.children)
        
        return [t.text for t in sorted(context_tokens, key=lambda x: x.i)]
    
    def analyze_medical_examples(self) -> None:
        """Analyze complex medical examples to demonstrate advanced context analysis."""
        medical_examples = [
            "The new chemotherapy treatment demonstrated excellent efficacy in reducing tumor size, "
            "though patients experienced mild side effects including fatigue and nausea.",
            
            "Despite initial concerns about the experimental drug, clinical trials revealed "
            "significant improvement in patient outcomes with minimal adverse reactions.",
            
            "The surgical procedure was complicated by unexpected bleeding, but the medical team "
            "successfully managed the situation, and the patient recovered well.",
            
            "Advanced imaging techniques enabled early detection of the malignancy, "
            "allowing for prompt intervention and improved prognosis.",
            
            "While the medication effectively controlled symptoms, long-term usage raised "
            "concerns about potential liver toxicity and required regular monitoring."
        ]
        
        keywords = ['treatment', 'patient', 'drug', 'procedure']
        
        for keyword in keywords:
            print(f"\n{'='*80}")
            print(f"ADVANCED ANALYSIS FOR KEYWORD: '{keyword.upper()}'")
            print(f"{'='*80}")
            
            for i, text in enumerate(medical_examples, 1):
                print(f"\nExample {i}: {text}")
                
                # Visualize dependency tree
                print(f"\nDependency Tree Analysis:")
                self.visualize_dependency_tree(text, keyword)
                
                # Compare context methods
                comparison = self.compare_context_methods(text, keyword)
                
                if comparison:
                    for result in comparison:
                        print(f"\nContext Comparison for '{keyword}' at position {result['keyword_position']}:")
                        print("-" * 60)
                        
                        print(f"Window Context ({result['context_sizes']['window']} tokens): "
                              f"{result['window_context']}")
                        
                        print(f"Semantic Cluster ({result['context_sizes']['semantic_cluster']} tokens): "
                              f"{list(result['semantic_context']['semantic_cluster'])}")
                        
                        print(f"Chunk Context ({result['context_sizes']['chunk']} tokens): "
                              f"{result['chunk_context']}")
                        
                        # Display semantic relationships
                        semantic = result['semantic_context']
                        if semantic['subject_relations']:
                            print(f"Subject Relations: {semantic['subject_relations']}")
                        if semantic['object_relations']:
                            print(f"Object Relations: {semantic['object_relations']}")
                        if semantic['modifier_relations']:
                            print(f"Modifier Relations: {semantic['modifier_relations']}")
                        if semantic['coordination_relations']:
                            print(f"Coordination Relations: {semantic['coordination_relations']}")


def main():
    """Demonstrate advanced dependency tree analysis."""
    print("Advanced Dependency Tree Analysis for Keyword Context Detection")
    print("=" * 70)
    
    try:
        analyzer = DependencyTreeAnalyzer()
        analyzer.analyze_medical_examples()
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nKey Features Demonstrated:")
        print("1. Dependency tree visualization")
        print("2. Semantic context extraction using syntactic relationships")
        print("3. Comparison of context window calculation methods")
        print("4. Advanced relationship detection (subject, object, modifier, coordination)")
        print("5. Semantic clustering through meaningful dependency traversal")
        
    except OSError as e:
        print(f"Error: {e}")
        print("Please install the required spaCy model:")
        print("python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    main()
