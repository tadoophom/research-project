import spacy
from keyword_context_analyzer import KeywordContextAnalyzer
from dependency_tree_analyzer import DependencyTreeAnalyzer
import pandas as pd
from typing import List, Dict
import json


class ComprehensiveNLPAnalysis:
    """
    Comprehensive NLP analysis combining keyword context detection with 
    advanced dependency tree analysis for improved context understanding.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize analyzers."""
        self.context_analyzer = KeywordContextAnalyzer(model_name)
        self.tree_analyzer = DependencyTreeAnalyzer(model_name)
        self.nlp = spacy.load(model_name)
    
    def get_medical_examples(self) -> List[str]:
        """Return comprehensive medical examples for analysis."""
        return [
            "The innovative cancer treatment protocol yielded outstanding results, with 85% of patients "
            "showing significant tumor reduction. However, some patients experienced severe side effects "
            "including neutropenia and peripheral neuropathy.",
            
            "Clinical studies demonstrate that the new antiviral medication effectively suppresses "
            "viral replication while maintaining excellent safety profiles. Patient compliance was "
            "high due to the simplified dosing regimen.",
            
            "The minimally invasive surgical procedure resulted in faster recovery times and reduced "
            "postoperative complications. Patients reported less pain and improved satisfaction scores "
            "compared to traditional open surgery techniques.",
            
            "Advanced molecular diagnostics enabled precise identification of genetic mutations, "
            "facilitating personalized treatment approaches. The targeted therapy showed remarkable "
            "efficacy in patients with specific biomarker profiles.",
            
            "Despite initial promising results, the experimental drug failed to meet primary endpoints "
            "in phase III trials. Investigators noted concerning safety signals requiring immediate "
            "study discontinuation and patient monitoring.",
            
            "Combination therapy with immunomodulatory agents and traditional chemotherapy produced "
            "synergistic effects, dramatically improving overall survival rates. The treatment regimen "
            "was well-tolerated with manageable adverse events.",
            
            "Breakthrough gene therapy techniques offer hope for patients with previously incurable "
            "genetic disorders. Early clinical data suggests durable therapeutic responses with "
            "minimal off-target effects.",
            
            "The diagnostic imaging protocol revealed subtle pathological changes that were missed "
            "by conventional methods. Early detection enabled timely intervention and prevented "
            "disease progression in high-risk patients."
        ]
    
    def perform_comprehensive_analysis(self, texts: List[str], keywords: List[str]) -> Dict:
        """
        Perform comprehensive analysis combining multiple approaches.
        """
        results = {
            'summary': {},
            'detailed_analysis': {},
            'method_comparison': {},
            'insights': {}
        }
        
        for keyword in keywords:
            print(f"\nProcessing keyword: '{keyword}'...")
            
            # Standard context analysis
            context_df = self.context_analyzer.analyze_text_collection(texts, keyword)
            context_summary = self.context_analyzer.generate_summary_report(context_df)
            
            # Dependency tree analysis
            dependency_results = []
            for text in texts:
                comparison = self.tree_analyzer.compare_context_methods(text, keyword)
                dependency_results.extend(comparison)
            
            # Store results
            results['summary'][keyword] = context_summary
            results['detailed_analysis'][keyword] = {
                'context_dataframe': context_df.to_dict('records') if not context_df.empty else [],
                'dependency_analysis': dependency_results
            }
            
            # Method comparison
            results['method_comparison'][keyword] = self._compare_methods(context_df, dependency_results)
            
            # Generate insights
            results['insights'][keyword] = self._generate_insights(context_df, dependency_results)
        
        return results
    
    def _compare_methods(self, context_df: pd.DataFrame, dependency_results: List[Dict]) -> Dict:
        """Compare different context analysis methods."""
        if context_df.empty or not dependency_results:
            return {'error': 'Insufficient data for comparison'}
        
        comparison = {
            'traditional_window': {
                'avg_sentiment': context_df['win_combined'].mean() if 'win_combined' in context_df.columns else 0,
                'sentiment_variance': context_df['win_combined'].var() if 'win_combined' in context_df.columns else 0,
            },
            'dependency_based': {
                'avg_sentiment': context_df['dep_combined'].mean() if 'dep_combined' in context_df.columns else 0,
                'sentiment_variance': context_df['dep_combined'].var() if 'dep_combined' in context_df.columns else 0,
            },
            'context_sizes': {
                'avg_window_size': sum(result['context_sizes']['window'] for result in dependency_results) / len(dependency_results),
                'avg_semantic_size': sum(result['context_sizes']['semantic_cluster'] for result in dependency_results) / len(dependency_results),
                'avg_chunk_size': sum(result['context_sizes']['chunk'] for result in dependency_results) / len(dependency_results),
            }
        }
        
        return comparison
    
    def _generate_insights(self, context_df: pd.DataFrame, dependency_results: List[Dict]) -> Dict:
        """Generate insights from the analysis."""
        insights = {
            'patterns': [],
            'recommendations': [],
            'statistical_observations': []
        }
        
        if not context_df.empty:
            # Sentiment distribution insights
            dep_positive = len(context_df[context_df['dep_combined'] > 0.1])
            dep_negative = len(context_df[context_df['dep_combined'] < -0.1])
            
            if dep_positive > dep_negative:
                insights['patterns'].append(f"Keyword appears predominantly in positive contexts ({dep_positive} vs {dep_negative})")
            elif dep_negative > dep_positive:
                insights['patterns'].append(f"Keyword appears predominantly in negative contexts ({dep_negative} vs {dep_positive})")
            else:
                insights['patterns'].append("Keyword appears in balanced positive/negative contexts")
            
            # Method accuracy insights
            correlation = context_df['dep_combined'].corr(context_df['win_combined']) if len(context_df) > 1 else 0
            insights['statistical_observations'].append(f"Correlation between dependency and window methods: {correlation:.3f}")
            
            if correlation < 0.5:
                insights['recommendations'].append("Low correlation suggests dependency-based analysis provides different insights than window-based")
            
        # Semantic relationship insights
        if dependency_results:
            total_semantic_relations = sum(
                len(result['semantic_context']['subject_relations']) +
                len(result['semantic_context']['object_relations']) +
                len(result['semantic_context']['modifier_relations'])
                for result in dependency_results
            )
            
            if total_semantic_relations > 0:
                insights['patterns'].append(f"Found {total_semantic_relations} semantic relationships providing rich context")
                insights['recommendations'].append("Semantic relationships provide valuable context beyond proximity-based methods")
        
        return insights
    
    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("COMPREHENSIVE NLP KEYWORD CONTEXT ANALYSIS REPORT")
        report.append("=" * 60)
        
        for keyword, data in results['summary'].items():
            if 'error' in data:
                continue
                
            report.append(f"\nKEYWORD: {keyword.upper()}")
            report.append("-" * 40)
            
            # Summary statistics
            report.append(f"Total occurrences: {data['total_occurrences']}")
            
            dep_analysis = data['dependency_analysis']
            report.append(f"\nDependency-based Analysis:")
            report.append(f"  Positive: {dep_analysis['positive']}")
            report.append(f"  Negative: {dep_analysis['negative']}")
            report.append(f"  Neutral: {dep_analysis['neutral']}")
            report.append(f"  Avg sentiment: {dep_analysis['avg_sentiment']:.3f}")
            
            # Method comparison
            if keyword in results['method_comparison']:
                comparison = results['method_comparison'][keyword]
                if 'error' not in comparison:
                    report.append(f"\nMethod Comparison:")
                    report.append(f"  Window-based avg sentiment: {comparison['traditional_window']['avg_sentiment']:.3f}")
                    report.append(f"  Dependency-based avg sentiment: {comparison['dependency_based']['avg_sentiment']:.3f}")
                    report.append(f"  Avg context sizes - Window: {comparison['context_sizes']['avg_window_size']:.1f}, "
                                f"Semantic: {comparison['context_sizes']['avg_semantic_size']:.1f}, "
                                f"Chunk: {comparison['context_sizes']['avg_chunk_size']:.1f}")
            
            # Insights
            if keyword in results['insights']:
                insights = results['insights'][keyword]
                if insights['patterns']:
                    report.append(f"\nKey Patterns:")
                    for pattern in insights['patterns']:
                        report.append(f"  • {pattern}")
                
                if insights['recommendations']:
                    report.append(f"\nRecommendations:")
                    for rec in insights['recommendations']:
                        report.append(f"  • {rec}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text
    
    def run_complete_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("Starting Comprehensive NLP Analysis...")
        print("=" * 50)
        
        # Get medical examples
        texts = self.get_medical_examples()
        keywords = ['treatment', 'patient', 'therapy', 'drug', 'procedure']
        
        print(f"Analyzing {len(texts)} medical texts for {len(keywords)} keywords...")
        
        # Perform analysis
        results = self.perform_comprehensive_analysis(texts, keywords)
        
        # Generate and display report
        report = self.generate_report(results, "analysis_report.txt")
        print("\n" + report)
        
        # Save detailed results as JSON
        with open("detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to detailed_results.json")
        
        print("\nAnalysis Features Demonstrated:")
        print("• Rule-based sentiment analysis with medical vocabulary")
        print("• TextBlob integration for sentiment scoring")
        print("• Dependency tree analysis for semantic relationships")
        print("• Multiple context extraction methods comparison")
        print("• Comprehensive reporting with insights and recommendations")
        print("• Statistical analysis of method effectiveness")


def main():
    """Main execution function."""
    try:
        analyzer = ComprehensiveNLPAnalysis()
        analyzer.run_complete_analysis()
        
    except OSError as e:
        print(f"Error: {e}")
        print("Please install the required spaCy model:")
        print("python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
