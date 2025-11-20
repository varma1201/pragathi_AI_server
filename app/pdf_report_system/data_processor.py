"""
Data Processor - Extracts insights from agent conversations
Processes raw validation data into structured report data
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class LogColors:
    """ANSI color codes for terminal output"""
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    BG_BLACK = '\033[40m'
    
    # Text colors
    BLACK = '\033[30m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def color_log(message, bg_color=LogColors.BG_YELLOW, text_color=LogColors.BLACK):
    """Wrap message with color codes"""
    return f"{bg_color}{text_color}{LogColors.BOLD}{message}{LogColors.RESET}"


class AgentDataProcessor:
    """Process agent evaluation data into structured report format"""
    
    def __init__(self, report_data: Dict[str, Any]):
        self.report_data = report_data
        # Try to get evaluated_data from multiple possible locations
        self.evaluated_data = (
            report_data.get('evaluated_data') or 
            report_data.get('raw_validation_result', {}).get('evaluated_data') or
            report_data.get('raw_validation_result', {}).get('evaluatedData') or
            report_data.get('validation_result', {}).get('evaluated_data') or
            report_data.get('detailed_analysis', {}).get('evaluated_data') or
            {}
        )
        
        logger.info(f"Evaluated data found: {bool(self.evaluated_data)}")
        if self.evaluated_data:
            logger.info(f"Evaluated data keys: {list(self.evaluated_data.keys())[:5]}...")
        
    def extract_all_agent_conversations(self) -> List[Dict[str, Any]]:
        """
        Extract ALL agent conversations with their insights
        Returns list of conversation objects with bullet points
        Handles both camelCase and snake_case field names
        """
        conversations = []
        
        if not self.evaluated_data or not isinstance(self.evaluated_data, dict):
            logger.warning("No evaluated_data found or invalid format")
            return conversations
        
        # Helper to get value with multiple key options
        def get_value(obj, *keys, default=None):
            for key in keys:
                if obj and key in obj:
                    return obj[key]
            return default
        
        for cluster_name, parameters in self.evaluated_data.items():
            if not isinstance(parameters, dict):
                continue
                
            for param_name, sub_params in parameters.items():
                if not isinstance(sub_params, dict):
                    continue
                    
                for sub_param_name, evaluation in sub_params.items():
                    if not isinstance(evaluation, dict):
                        continue
                    
                    # Handle both camelCase and snake_case
                    score = get_value(evaluation, 'assigned_score', 'assignedScore', 'score', default=0)
                    try:
                        score = float(score) if score else 0
                    except (ValueError, TypeError):
                        score = 0
                    
                    conversation = {
                        'cluster': cluster_name,
                        'parameter': param_name,
                        'sub_parameter': sub_param_name,
                        'score': score,
                        'explanation': get_value(evaluation, 'explanation', 'Explanation', default=''),
                        'strengths': get_value(evaluation, 'strengths', 'Strengths', default=[]) or [],
                        'weaknesses': get_value(evaluation, 'weaknesses', 'Weaknesses', default=[]) or [],
                        'key_insights': get_value(evaluation, 'key_insights', 'keyInsights', 'insights', default=[]) or [],
                        'recommendations': get_value(evaluation, 'recommendations', 'Recommendations', default=[]) or [],
                        'risk_factors': get_value(evaluation, 'risk_factors', 'riskFactors', default=[]) or [],
                        'assumptions': get_value(evaluation, 'assumptions', 'Assumptions', default=[]) or [],
                        'agent_id': get_value(evaluation, 'agent_id', 'agentId', default='')
                    }
                    conversations.append(conversation)
        
        logger.info(f"Extracted {len(conversations)} agent conversations from {len(self.evaluated_data)} clusters")
        return conversations
    
    def group_by_cluster(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by cluster"""
        grouped = {}
        for conv in conversations:
            cluster = conv['cluster']
            if cluster not in grouped:
                grouped[cluster] = []
            grouped[cluster].append(conv)
        return grouped
    
    def group_by_parameter(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by parameter within a cluster"""
        grouped = {}
        for conv in conversations:
            key = f"{conv['cluster']}::{conv['parameter']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(conv)
        return grouped
    
    def calculate_cluster_scores(self, conversations: List[Dict]) -> Dict[str, float]:
        """Calculate average score for each cluster"""
        cluster_scores = {}
        cluster_counts = {}
        
        for conv in conversations:
            cluster = conv['cluster']
            score = conv['score']
            
            if cluster not in cluster_scores:
                cluster_scores[cluster] = 0
                cluster_counts[cluster] = 0
            
            cluster_scores[cluster] += score
            cluster_counts[cluster] += 1
        
        # Calculate averages
        for cluster in cluster_scores:
            if cluster_counts[cluster] > 0:
                cluster_scores[cluster] = cluster_scores[cluster] / cluster_counts[cluster]
        
        return cluster_scores
    
    def extract_strengths_and_weaknesses(self, conversations: List[Dict]) -> Dict[str, List[str]]:
        """Extract all strengths and weaknesses from agent conversations"""
        all_strengths = []
        all_weaknesses = []
        
        for conv in conversations:
            # High-scoring items are strengths
            if conv['score'] >= 70:
                for strength in conv['strengths']:
                    all_strengths.append({
                        'text': strength,
                        'cluster': conv['cluster'],
                        'parameter': conv['sub_parameter'],
                        'score': conv['score']
                    })
            
            # Low-scoring items are weaknesses
            if conv['score'] < 60:
                for weakness in conv['weaknesses']:
                    all_weaknesses.append({
                        'text': weakness,
                        'cluster': conv['cluster'],
                        'parameter': conv['sub_parameter'],
                        'score': conv['score'],
                        'severity': self._get_severity(conv['score'])
                    })
        
        return {
            'strengths': sorted(all_strengths, key=lambda x: x['score'], reverse=True),
            'weaknesses': sorted(all_weaknesses, key=lambda x: x['score'])
        }
    
    def _get_severity(self, score: float) -> str:
        """Determine severity level based on score"""
        if score < 30:
            return 'Critical'
        elif score < 50:
            return 'High'
        else:
            return 'Moderate'
    
    def extract_recommendations(self, conversations: List[Dict]) -> List[Dict[str, Any]]:
        """Extract all recommendations from agents"""
        all_recommendations = []
        
        for conv in conversations:
            for rec in conv['recommendations']:
                all_recommendations.append({
                    'text': rec,
                    'cluster': conv['cluster'],
                    'parameter': conv['sub_parameter'],
                    'priority': 'High' if conv['score'] < 50 else 'Medium' if conv['score'] < 70 else 'Low'
                })
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        all_recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return all_recommendations
    
    def generate_cluster_summary(self, cluster_name: str, conversations: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive summary for a cluster"""
        cluster_convs = [c for c in conversations if c['cluster'] == cluster_name]
        
        if not cluster_convs:
            return {}
        
        # Calculate cluster score
        avg_score = sum(c['score'] for c in cluster_convs) / len(cluster_convs)
        
        # Get strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for conv in cluster_convs:
            if conv['score'] >= 70:
                strengths.extend(conv['strengths'])
            if conv['score'] < 60:
                weaknesses.extend(conv['weaknesses'])
        
        # Get all insights
        all_insights = []
        for conv in cluster_convs:
            all_insights.extend(conv['key_insights'])
        
        return {
            'cluster_name': cluster_name,
            'overall_score': avg_score,
            'status': self._get_status(avg_score),
            'num_parameters': len(cluster_convs),
            'strengths': strengths[:5],  # Top 5
            'weaknesses': weaknesses[:5],  # Top 5
            'key_insights': all_insights[:5],  # Top 5
            'parameters': cluster_convs
        }
    
    def _get_status(self, score: float) -> str:
        """Get status label for score"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Weak'
        else:
            return 'Poor'
    
    def process_complete_report_data(self) -> Dict[str, Any]:
        """
        Process all data and return complete structure for PDF/AI report generation.
        1) Uses existing agent conversations if present (preferred).
        2) If absent, converts detailed_analysis.cluster_analyses → conversations.
        """
        import logging
        logger = logging.getLogger(__name__)

        def _convert_cluster_analyses_to_conversations(cluster_analyses: Dict) -> List[Dict]:
            """
            Convert cluster_analyses to the normalized 'conversations' format:
            Each item contains: cluster, parameter, sub_parameter, score, explanation, strengths, weaknesses
            Supports categories: strong, moderate, weak in each cluster's parameters.
            """
            if not isinstance(cluster_analyses, dict):
                logger.warning("cluster_analyses is not a dict; skipping conversion")
                return []

            conversations: List[Dict] = []
            for cluster_name, cluster_data in cluster_analyses.items():
                params = (cluster_data or {}).get('parameters', {})
                # Categories often: strong, moderate, weak
                for category in ['strong', 'moderate', 'weak']:
                    for item in params.get(category, []) or []:
                        # item.name may be like "Innovation Factor > Novelty of Solution"
                        raw_name = (item or {}).get('name', 'Unknown > Unknown')
                        if isinstance(raw_name, str):
                            parts = [p.strip() for p in raw_name.split('>')]
                        else:
                            parts = ['Unknown', 'Unknown']

                        parameter = parts[0] if len(parts) > 0 else 'Unknown'
                        sub_parameter = parts[1] if len(parts) > 1 else (parts[0] if parts else 'Unknown')

                        score_val = item.get('score', 0)
                        try:
                            score = float(score_val)
                        except Exception:
                            score = 0.0

                        conversation = {
                            'cluster': str(cluster_name or 'Unknown'),
                            'parameter': str(parameter or 'General'),
                            'sub_parameter': str(sub_parameter or 'Metric'),
                            'score': score,
                            'explanation': item.get('explanation', '') or '',
                            'strengths': item.get('strengths', []) or [],
                            'weaknesses': item.get('weaknesses', []) or [],
                        }
                        conversations.append(conversation)
            return conversations
        
        logger.info(color_log("=" * 80, LogColors.BG_CYAN))
        logger.info(color_log("🔍 DEBUG: process_complete_report_data called", LogColors.BG_YELLOW))
        logger.info(f"Report data keys: {list(self.report_data.keys())}")
        # 1) Preferred: existing conversations from agents
        conversations = self.extract_all_agent_conversations() or []

        logger.info(f"Agent conversations found: {len(conversations)}")

        # 2) If no conversations, try to convert from detailed_analysis.cluster_analyses
        if not conversations:
            logger.warning("No agent conversations found; attempting conversion from detailed_analysis.cluster_analyses")
            raw = self.report_data or {}
            # Try multiple casings/keys for robustness
            detailed_analysis = self.report_data.get('detailed_analysis', {})
            
            cluster_analyses = (
                detailed_analysis.get('cluster_analyses')
                or detailed_analysis.get('clusterAnalyses')
                or {}
            )
            conversations = _convert_cluster_analyses_to_conversations(cluster_analyses)

        # 3) Final sanity normalization of conversation items
        normalized: List[Dict] = []
        for c in conversations:
            if not isinstance(c, dict):
                continue
            try:
                score_val = c.get('score', 0)
                score = float(score_val)
            except Exception:
                score = 0.0

            normalized.append({
                'cluster': str(c.get('cluster', 'Unknown')),
                'parameter': str(c.get('parameter') or c.get('param') or 'General'),
                'sub_parameter': str(c.get('sub_parameter') or c.get('subParam') or c.get('question') or 'Metric'),
                'score': score,
                'explanation': c.get('explanation', '') or '',
                'strengths': c.get('strengths', []) or [],
                'weaknesses': c.get('weaknesses', []) or [],
            })

        if not normalized:
            logger.warning("No conversations available after conversion; returning empty dict")
            return {}

        # Group, score, and summarize using existing helpers
        grouped_by_cluster = self.group_by_cluster(normalized)
        cluster_scores = self.calculate_cluster_scores(normalized)
        strengths_weaknesses = self.extract_strengths_and_weaknesses(normalized)
        recommendations = self.extract_recommendations(normalized)

        cluster_summaries: Dict[str, Any] = {}
        for cluster_name in grouped_by_cluster.keys():
            try:
                cluster_summaries[cluster_name] = self.generate_cluster_summary(cluster_name, normalized)
            except Exception as e:
                logger.error(f"Failed to generate cluster summary for {cluster_name}: {e}")
                cluster_summaries[cluster_name] = {}

        # Metadata fallbacks
        raw = self.report_data or {}
        overall_score = (
            raw.get('overall_score')
            or (raw.get('raw_validation_result') or {}).get('overall_score')
            or 0
        )
        validation_outcome = (
            raw.get('validation_outcome')
            or (raw.get('raw_validation_result') or {}).get('validation_outcome')
            or ''
        )

        return {
            'metadata': {
                'title': raw.get('title', 'Validation Report'),
                'user_id': raw.get('user_id', ''),
                'report_id': str(raw.get('_id', '')),
                'created_at': raw.get('created_at', ''),
                'overall_score': overall_score,
                'validation_outcome': validation_outcome,
                'total_agents': len(normalized),
            },
            'cluster_scores': cluster_scores,
            'cluster_summaries': cluster_summaries,
            'strengths': strengths_weaknesses.get('strengths', []),
            'weaknesses': strengths_weaknesses.get('weaknesses', []),
            'recommendations': recommendations,
            'all_conversations': normalized,
        }


