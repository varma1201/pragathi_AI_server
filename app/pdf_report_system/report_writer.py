# Enhanced AI Report Writer - Generates Complete UI Data Structures

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Tuple
import json
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class AIReportWriter:
    """
    AI that reads agent conversations and writes comprehensive reports for UI consumption
    Generates ALL data structures needed by the React frontend
    """
    
    def __init__(self, progress_callback=None):
        """
        Initialize AI Report Writer
        
        Args:
            progress_callback: Optional function to call with progress updates
                              Signature: callback(message: str, progress: float)
        """
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4-mini",
            max_tokens=4000,
            timeout=120
        )
        self.progress_callback = progress_callback
    
    def _update_progress(self, message: str, progress: float):
        """Send progress update if callback is provided"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def write_comprehensive_report(self, agent_conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Main method: Generate complete report data for the React UI
        
        Args:
            agent_conversations: List of all agent evaluations with their insights
            metadata: Report metadata (title, score, validation_outcome, etc.)
        
        Returns:
            Complete report with ALL sections needed by the UI including:
            - roadmap (project phases)
            - highlights_lowlights (sorted top/bottom performers)
            - action_points (conditional next steps)
            - And all existing analytical sections
        """
        total_steps = 10 + len(set(c.get('cluster', 'Unknown') for c in agent_conversations))
        current_step = 0
        
        logger.info(f"AI Report Writer analyzing {len(agent_conversations)} agent conversations...")
        self._update_progress("📊 Analyzing agent conversations...", 0)
        
        # Group conversations by cluster
        clustered_data = self._group_by_cluster(agent_conversations)
        current_step += 1
        self._update_progress(f"📋 Grouped conversations into {len(clustered_data)} clusters", 
                            current_step / total_steps * 100)
        
        # Write report for each cluster (detailed)
        cluster_reports = {}
        for idx, (cluster_name, conversations) in enumerate(clustered_data.items(), 1):
            logger.info(f"Writing detailed analysis for {cluster_name}...")
            self._update_progress(f"✍️ Writing {cluster_name} analysis ({idx}/{len(clustered_data)})...", 
                                (current_step / total_steps) * 100)
            cluster_reports[cluster_name] = self._write_cluster_report(
                cluster_name, conversations, metadata
            )
            current_step += 1
        
        # Write executive summary
        self._update_progress("📝 Writing Executive Summary...", (current_step / total_steps) * 100)
        executive_summary = self._write_executive_summary(
            cluster_reports, agent_conversations, metadata
        )
        current_step += 1
        
        # Write TAM/SAM/SOM analysis
        self._update_progress("📊 Analyzing Market Size (TAM/SAM/SOM)...", (current_step / total_steps) * 100)
        market_analysis = self._write_market_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write TRL analysis
        self._update_progress("🔬 Analyzing Technology Readiness (TRL)...", (current_step / total_steps) * 100)
        trl_analysis = self._write_trl_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write comprehensive pros and cons
        self._update_progress("⚖️ Analyzing Pros and Cons...", (current_step / total_steps) * 100)
        pros_cons = self._write_pros_cons_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write detailed weaknesses analysis
        self._update_progress("🔍 Analyzing Weaknesses...", (current_step / total_steps) * 100)
        weaknesses_analysis = self._write_weaknesses_analysis(
            agent_conversations, metadata
        )
        current_step += 1
        
        # Write conclusion
        self._update_progress("📋 Writing Conclusion...", (current_step / total_steps) * 100)
        conclusion = self._write_conclusion(
            cluster_reports, metadata, market_analysis, trl_analysis
        )
        current_step += 1
        
        # Generate roadmap phases (NEW)
        self._update_progress("🗺️ Generating Project Roadmap...", (current_step / total_steps) * 100)
        roadmap = self._write_roadmap_phases(trl_analysis, metadata)
        current_step += 1
        
        # Generate detailed viability assessment
        self._update_progress("📊 Generating Detailed Viability Assessment...", (current_step / total_steps) * 100)
        detailed_viability_assessment = self._write_detailed_viability_assessment(agent_conversations, metadata)
        current_step += 1
        
        # Generate highlights & lowlights (NEW)
        self._update_progress("⭐ Generating Highlights & Lowlights...", (current_step / total_steps) * 100)
        highlights_lowlights = self._write_highlights_lowlights(detailed_viability_assessment)
        current_step += 1
        
        # Generate risk analysis
        self._update_progress("🚨 Writing Explicit Risk Analysis...", (current_step / total_steps) * 100)
        risk_analysis = self._write_explicit_risk_analysis(
            agent_conversations, metadata
        )
        current_step += 1       
        
        # Generate benchmarking
        self._update_progress("📊 Writing Benchmarking Analysis...", (current_step / total_steps) * 100)
        benchmarking = self._write_benchmarking_analysis(
            agent_conversations, metadata
        )
        current_step += 1       
        
        # Generate structured recommendations
        self._update_progress("📋 Writing Structured Recommendations...", (current_step / total_steps) * 100)
        structured_recommendations = self._write_structured_recommendations(
            conclusion, market_analysis, trl_analysis, metadata
        )
        current_step += 1
        
        # Generate action points (NEW - depends on highlights)
        self._update_progress("📋 Generating Action Points...", (current_step / total_steps) * 100)
        action_points = self._write_action_points(
            conclusion, metadata, highlights_lowlights['bottomPerformers']
        )
        current_step += 1
        
        self._update_progress("✅ Report writing complete!", 100)
        
        return {
            'executive_summary': executive_summary,
            'cluster_reports': cluster_reports,
            'market_analysis': market_analysis,
            'trl_analysis': trl_analysis,
            'pros_cons': pros_cons,
            'weaknesses_analysis': weaknesses_analysis,
            'conclusion': conclusion,
            'risk_analysis': risk_analysis,
            'benchmarking': benchmarking,
            'structured_recommendations': structured_recommendations,
            'detailed_viability_assessment': detailed_viability_assessment,
            'roadmap': roadmap,  # NEW - for Project Roadmap UI
            'highlights_lowlights': highlights_lowlights,  # NEW - for Highlights & Lowlights UI
            'action_points': action_points,  # NEW - for Next Steps UI
            'metadata': metadata
        }
    
    def _group_by_cluster(self, conversations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group conversations by cluster"""
        grouped = {}
        for conv in conversations:
            cluster = conv.get('cluster', 'Unknown')
            if cluster not in grouped:
                grouped[cluster] = []
            grouped[cluster].append(conv)
        return grouped
    
    def _write_cluster_report(self, cluster_name: str, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive analysis for a single cluster"""
        context = self._prepare_cluster_context(cluster_name, conversations)
        
        prompt = f"""You are a senior business analyst writing a comprehensive validation report. You have access to detailed evaluations from {len(conversations)} expert agents who analyzed the "{metadata['title']}" startup idea.

**Your Task**: Write a detailed, professional analysis of the **{cluster_name}** category based on the expert discussions below.

**Expert Agent Conversations**:
{context}

**Write a comprehensive analysis with the following structure**:

1. **Overview** (2-3 bullet points summarizing the cluster)
2. **Detailed Parameter Analysis** (for EACH parameter, write):
   • Parameter name and score
   • Key findings from experts (bullet points)
   • Strengths identified (bullet points if score > 70)
   • Weaknesses identified (bullet points if score < 60)
   • Expert recommendations (bullet points)

3. **Cluster Summary** (2-3 bullet points on overall cluster performance)

**Important Guidelines**:
- Use ONLY bullet points, NO long paragraphs
- Base everything on the expert conversations provided
- Include specific scores mentioned by experts
- Highlight both positive and negative aspects
- Be objective and analytical
- Each bullet point should be ONE clear statement
- Use professional business language

Return ONLY a JSON object with this structure:
{{
  "overview": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "parameters": [
    {{
      "name": "parameter name",
      "score": 75.0,
      "findings": ["finding 1", "finding 2"],
      "strengths": ["strength 1", "strength 2"],
      "weaknesses": ["weakness 1", "weakness 2"],
      "recommendations": ["recommendation 1", "recommendation 2"]
    }}
  ],
  "cluster_summary": ["summary point 1", "summary point 2", "summary point 3"]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            report = json.loads(content)
            
            # Calculate cluster score
            scores = [p['score'] for p in report['parameters'] if 'score' in p]
            avg_score = sum(scores) / len(scores) if scores else 0
            report['cluster_score'] = avg_score
            report['cluster_name'] = cluster_name
            
            logger.info(f"✅ Completed {cluster_name} analysis: {len(report['parameters'])} parameters")
            return report
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing cluster report: {e}")
            return self._create_fallback_cluster_report(cluster_name, conversations)
    
    def _prepare_cluster_context(self, cluster_name: str, conversations: List[Dict]) -> str:
        """Prepare formatted context from agent conversations"""
        context_parts = []
        
        for i, conv in enumerate(conversations, 1):
            context_parts.append(f"\n--- Expert {i}: {conv.get('sub_parameter', 'Unknown')} Specialist ---")
            context_parts.append(f"Score: {conv.get('score', 0):.1f}/100")
            context_parts.append(f"Assessment: {conv.get('explanation', 'No explanation provided')}")
            
            if conv.get('strengths'):
                context_parts.append("Strengths:")
                for s in conv['strengths']:
                    context_parts.append(f"  • {s}")
            
            if conv.get('weaknesses'):
                context_parts.append("Weaknesses:")
                for w in conv['weaknesses']:
                    context_parts.append(f"  • {w}")
            
            if conv.get('key_insights'):
                context_parts.append("Key Insights:")
                for insight in conv['key_insights']:
                    context_parts.append(f"  • {insight}")
            
            if conv.get('recommendations'):
                context_parts.append("Recommendations:")
                for rec in conv['recommendations']:
                    context_parts.append(f"  • {rec}")
        
        return "\n".join(context_parts)
    
    def _write_executive_summary(self, cluster_reports: Dict, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write executive summary based on all cluster reports"""
        all_scores = [conv['score'] for conv in conversations]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        strengths = []
        weaknesses = []
        
        for conv in conversations:
            if conv['score'] >= 75:
                for s in conv.get('strengths', []):
                    strengths.append({'text': s, 'score': conv['score'], 'area': conv['sub_parameter']})
            if conv['score'] < 55:
                for w in conv.get('weaknesses', []):
                    weaknesses.append({'text': w, 'score': conv['score'], 'area': conv['sub_parameter']})
        
        strengths.sort(key=lambda x: x['score'], reverse=True)
        weaknesses.sort(key=lambda x: x['score'])
        
        prompt = f"""You are writing the Executive Summary for a startup validation report for "{metadata['title']}".

**Overall Score**: {avg_score:.1f}/100
**Validation Outcome**: {metadata['validation_outcome']}

**Top 10 Strengths** (from expert agents):
{self._format_items_for_prompt(strengths[:10])}

**Top 10 Critical Weaknesses** (from expert agents):
{self._format_items_for_prompt(weaknesses[:10])}

**Cluster Performance**:
{self._format_cluster_scores(cluster_reports)}

**Write an Executive Summary with**:
1. **Key Findings** (5-7 bullet points summarizing overall assessment)
2. **Major Strengths** (5-6 bullet points from the strengths above)
3. **Critical Concerns** (5-6 bullet points from the weaknesses above)
4. **Strategic Recommendations** (5-6 bullet points for immediate actions)

Use ONLY bullet points. Be concise and impactful.

Return JSON:
{{
  "key_findings": ["finding 1", "finding 2", ...],
  "major_strengths": ["strength 1", "strength 2", ...],
  "critical_concerns": ["concern 1", "concern 2", ...],
  "strategic_recommendations": ["recommendation 1", "recommendation 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            summary = json.loads(content)
            summary['overall_score'] = avg_score
            summary['outcome'] = metadata['validation_outcome']
            
            return summary
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing executive summary: {e}")
            return self._create_fallback_summary(strengths, weaknesses, avg_score, metadata)
    
    def _write_market_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive TAM/SAM/SOM analysis"""
        market_conversations = [c for c in conversations if 'market' in c.get('cluster', '').lower() or 
                              'market' in c.get('sub_parameter', '').lower()]
        
        prompt = f"""You are a market analyst writing a comprehensive market size analysis for "{metadata['title']}".

**Expert Evaluations**:
{self._prepare_cluster_context('Market Analysis', market_conversations if market_conversations else conversations[:10])}

**Write a detailed market analysis with**:

1. **TAM (Total Addressable Market)**:
   - Definition: Total market demand for this product/service globally
   - Estimated size (in USD/INR)
   - Growth rate and trends
   - Key assumptions (3-4 bullet points)

2. **SAM (Serviceable Available Market)**:
   - Definition: Portion of TAM that can be realistically served
   - Estimated size (in USD/INR)
   - Geographic and demographic focus
   - Market accessibility factors (3-4 bullet points)

3. **SOM (Serviceable Obtainable Market)**:
   - Definition: Portion of SAM that can be captured in 3-5 years
   - Estimated size (in USD/INR)
   - Market share assumptions
   - Competitive landscape considerations (3-4 bullet points)

4. **Market Opportunity Summary** (4-5 bullet points)

**Important**:
- Use ONLY bullet points
- Base estimates on expert evaluations
- Include Indian market context
- Be realistic and data-driven
- All numbers should be justified

Return JSON:
{{
  "tam": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "growth_rate": "Growth percentage",
    "assumptions": ["assumption 1", "assumption 2", ...],
    "trends": ["trend 1", "trend 2", ...]
  }},
  "sam": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "geographic_focus": "Primary markets",
    "accessibility_factors": ["factor 1", "factor 2", ...],
    "demographics": ["demographic 1", "demographic 2", ...]
  }},
  "som": {{
    "definition": "Brief definition",
    "size": "Estimated size with unit",
    "market_share": "Target market share %",
    "competitive_landscape": ["consideration 1", "consideration 2", ...],
    "capture_strategy": ["strategy 1", "strategy 2", ...]
  }},
  "opportunity_summary": ["summary point 1", "summary point 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Market analysis (TAM/SAM/SOM) completed")
            return analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing market analysis: {e}")
            return self._create_fallback_market_analysis(metadata)
    
    def _write_trl_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write Technology Readiness Level (TRL) analysis with timeline"""
        tech_conversations = [c for c in conversations if 'execution' in c.get('cluster', '').lower() or 
                            'technology' in c.get('sub_parameter', '').lower() or
                            'technical' in c.get('sub_parameter', '').lower()]
        
        prompt = f"""You are a technology analyst writing a TRL (Technology Readiness Level) analysis for "{metadata['title']}".

**Expert Evaluations on Technology & Execution**:
{self._prepare_cluster_context('Technology Analysis', tech_conversations if tech_conversations else conversations[:10])}

**TRL Levels (1-9)**:
- TRL 1: Basic principles observed
- TRL 2: Technology concept formulated
- TRL 3: Experimental proof of concept
- TRL 4: Technology validated in lab
- TRL 5: Technology validated in relevant environment
- TRL 6: Technology demonstrated in relevant environment
- TRL 7: System prototype demonstration in operational environment
- TRL 8: System complete and qualified
- TRL 9: Actual system proven in operational environment

**Write a comprehensive TRL analysis with**:

1. **Current TRL Assessment**:
   - Current TRL level (1-9)
   - Justification (3-4 bullet points)
   - Key technology components status

2. **TRL Progression Timeline**:
   - TRL 1-3: Timeline and milestones (2-3 bullet points)
   - TRL 4-6: Timeline and milestones (2-3 bullet points)
   - TRL 7-9: Timeline and milestones (2-3 bullet points)
   - Estimated time to market readiness

3. **Technology Risks & Challenges** (4-5 bullet points)

4. **Technology Strengths** (3-4 bullet points)

5. **Recommendations for TRL Advancement** (4-5 bullet points)

**Important**:
- Use ONLY bullet points
- Base assessment on expert evaluations
- Be realistic about timelines
- Include Indian market technology context

Return JSON:
{{
  "current_trl": {{
    "level": 5,
    "justification": ["justification 1", "justification 2", ...],
    "components_status": ["component 1 status", "component 2 status", ...]
  }},
  "timeline": {{
    "trl_1_3": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "trl_4_6": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "trl_7_9": {{
      "timeframe": "X months",
      "milestones": ["milestone 1", "milestone 2", ...]
    }},
    "time_to_market": "Estimated time"
  }},
  "risks": ["risk 1", "risk 2", ...],
  "strengths": ["strength 1", "strength 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ TRL analysis completed")
            return analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing TRL analysis: {e}")
            return self._create_fallback_trl_analysis(metadata)
    
    def _write_pros_cons_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive pros and cons analysis"""
        all_pros = []
        all_cons = []
        
        for conv in conversations:
            score = conv.get('score', 0)
            area = conv.get('sub_parameter', 'Unknown')
            
            if score >= 70:
                for strength in conv.get('strengths', []):
                    all_pros.append({'text': strength, 'score': score, 'area': area})
                if conv.get('key_insights'):
                    for insight in conv['key_insights']:
                        if 'strong' in insight.lower() or 'positive' in insight.lower():
                            all_pros.append({'text': insight, 'score': score, 'area': area})
            
            if score < 60:
                for weakness in conv.get('weaknesses', []):
                    all_cons.append({'text': weakness, 'score': score, 'area': area})
                if conv.get('risk_factors'):
                    for risk in conv['risk_factors']:
                        all_cons.append({'text': risk, 'score': score, 'area': area})
        
        all_pros.sort(key=lambda x: x['score'], reverse=True)
        all_cons.sort(key=lambda x: x['score'])
        
        prompt = f"""You are an analyst synthesizing pros and cons for "{metadata['title']}".

**Overall Score**: {metadata['overall_score']:.1f}/100

**Top 15 Strengths** (from expert evaluations):
{self._format_items_for_prompt(all_pros[:15])}

**Top 15 Weaknesses** (from expert evaluations):
{self._format_items_for_prompt(all_cons[:15])}

**Write a comprehensive pros and cons analysis**:

1. **Major Advantages** (8-10 bullet points covering):
   - Market opportunities
   - Technology strengths
   - Business model advantages
   - Competitive advantages
   - Scalability potential

2. **Key Disadvantages** (8-10 bullet points covering):
   - Market challenges
   - Technology gaps
   - Business model concerns
   - Competitive threats
   - Execution risks

3. **Balanced Assessment** (4-5 bullet points weighing pros vs cons)

**Important**:
- Use ONLY bullet points
- Be specific and actionable
- Base on expert evaluations
- Include Indian market context

Return JSON:
{{
  "advantages": ["advantage 1", "advantage 2", ...],
  "disadvantages": ["disadvantage 1", "disadvantage 2", ...],
  "balanced_assessment": ["assessment point 1", "assessment point 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Pros and cons analysis completed")
            return analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing pros/cons analysis: {e}")
            return self._create_fallback_pros_cons(all_pros, all_cons)
    
    def _write_weaknesses_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write detailed weaknesses analysis with severity and recommendations"""
        all_weaknesses = []
        for conv in conversations:
            score = conv.get('score', 0)
            area = conv.get('sub_parameter', 'Unknown')
            cluster = conv.get('cluster', 'Unknown')
            
            if score < 60:
                for weakness in conv.get('weaknesses', []):
                    severity = "Critical" if score < 40 else "High" if score < 50 else "Moderate"
                    all_weaknesses.append({
                        'text': weakness,
                        'score': score,
                        'area': area,
                        'cluster': cluster,
                        'severity': severity
                    })
        
        all_weaknesses.sort(key=lambda x: (x['score'], x['severity'] == 'Critical', x['severity'] == 'High'))
        
        prompt = f"""You are an analyst writing a detailed weaknesses analysis for "{metadata['title']}".

**Overall Score**: {metadata['overall_score']:.1f}/100

**All Identified Weaknesses** (from expert evaluations):
{self._format_weaknesses_for_prompt(all_weaknesses)}

**Write a comprehensive weaknesses analysis**:

1. **Critical Weaknesses** (Score < 40/100):
   - List all critical weaknesses (4-6 bullet points)
   - Impact assessment for each
   - Immediate action required

2. **High Priority Weaknesses** (Score 40-50/100):
   - List high priority weaknesses (5-7 bullet points)
   - Impact assessment
   - Short-term action plan

3. **Moderate Weaknesses** (Score 50-60/100):
   - List moderate weaknesses (4-6 bullet points)
   - Impact assessment
   - Medium-term improvement plan

4. **Weakness Patterns** (3-4 bullet points identifying common themes)

5. **Remediation Strategy** (5-6 bullet points on addressing weaknesses)

**Important**:
- Use ONLY bullet points
- Be specific about each weakness
- Include actionable remediation steps
- Prioritize by severity

Return JSON:
{{
  "critical": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "high_priority": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "moderate": [
    {{"weakness": "weakness text", "impact": "impact description", "action": "action required"}},
    ...
  ],
  "patterns": ["pattern 1", "pattern 2", ...],
  "remediation_strategy": ["strategy 1", "strategy 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            analysis = json.loads(content)
            logger.info("✅ Weaknesses analysis completed")
            return analysis
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing weaknesses analysis: {e}")
            return self._create_fallback_weaknesses(all_weaknesses)
    
    def _write_conclusion(self, cluster_reports: Dict, metadata: Dict, market_analysis: Dict = None, trl_analysis: Dict = None) -> Dict[str, Any]:
        """Write final conclusion and verdict with market and TRL context"""
        score = metadata['overall_score']
        
        market_context = ""
        if market_analysis:
            market_context = f"""
**Market Analysis Summary**:
- TAM: {market_analysis.get('tam', {}).get('size', 'N/A')}
- SAM: {market_analysis.get('sam', {}).get('size', 'N/A')}
- SOM: {market_analysis.get('som', {}).get('size', 'N/A')}
"""
        
        trl_context = ""
        if trl_analysis:
            trl_context = f"""
**Technology Readiness**:
- Current TRL: {trl_analysis.get('current_trl', {}).get('level', 'N/A')}
- Time to Market: {trl_analysis.get('timeline', {}).get('time_to_market', 'N/A')}
"""
        
        prompt = f"""Write a comprehensive conclusion for a startup validation report for "{metadata['title']}".

**Overall Score**: {score:.1f}/100

**Cluster Summary**:
{self._format_cluster_scores(cluster_reports)}
{market_context}
{trl_context}

**Write**:
1. **Final Verdict** (3-4 bullet points on investment recommendation considering market size and technology readiness)
2. **Path Forward** (5-6 bullet points on next steps including TRL progression)
3. **Success Factors** (4-5 bullet points on what's needed for success)
4. **Risk Mitigation** (4-5 bullet points on managing key risks)
5. **Market Opportunity Assessment** (3-4 bullet points on TAM/SAM/SOM potential)

Use bullet points only. Be decisive and actionable. Integrate market and technology insights.

Return JSON:
{{
  "final_verdict": ["verdict point 1", "verdict point 2", ...],
  "path_forward": ["next step 1", "next step 2", ...],
  "success_factors": ["factor 1", "factor 2", ...],
  "risk_mitigation": ["mitigation 1", "mitigation 2", ...],
  "market_assessment": ["assessment 1", "assessment 2", ...]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            conclusion = json.loads(content)
            conclusion['investment_decision'] = self._get_investment_decision(score)
            
            return conclusion
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing conclusion: {e}")
            return self._create_fallback_conclusion(score, metadata)
    
    def _format_items_for_prompt(self, items: List[Dict]) -> str:
        """Format items for LLM prompt"""
        lines = []
        for item in items:
            lines.append(f"• [{item['score']:.0f}/100] {item['area']}: {item['text']}")
        return "\n".join(lines) if lines else "None identified"
    
    def _format_cluster_scores(self, cluster_reports: Dict) -> str:
        """Format cluster scores for prompt"""
        lines = []
        for name, report in cluster_reports.items():
            score = report.get('cluster_score', 0)
            lines.append(f"• {name}: {score:.1f}/100")
        return "\n".join(lines) if lines else "No cluster data"
    
    def _get_investment_decision(self, score: float) -> str:
        """Get investment recommendation - strict evaluation based on expert consensus"""
        if score >= 75:
            return "STRONG YES - Recommended for investment"
        elif score >= 65:
            return "YES WITH CONDITIONS - Proceed with improvements"
        elif score >= 50:
            return "MAYBE - Significant concerns to address"
        else:
            return "NO - NOT RECOMMENDED - Fundamental issues present"
    
    # NEW METHOD: Generate Project Roadmap
    def _write_roadmap_phases(self, trl_analysis: Dict, metadata: Dict) -> Dict[str, Any]:
        """
        Generate complete roadmap structure for UI
        Returns phases with TRL ranges, timelines, and milestones
        """
        current_trl = trl_analysis.get('current_trl', {}).get('level', 3)
        timeline = trl_analysis.get('timeline', {})
        
        phases = [
            {
                "name": "Concept & Research",
                "trls": [1, 2, 3],
                "timeline": timeline.get('trl_1_3', {}).get('timeframe', '3-6 months'),
                "milestones": timeline.get('trl_1_3', {}).get('milestones', [
                    "Basic principles validated",
                    "Technology concept formulated",
                    "Experimental proof of concept"
                ])
            },
            {
                "name": "Development & Validation",
                "trls": [4, 5, 6],
                "timeline": timeline.get('trl_4_6', {}).get('timeframe', '6-12 months'),
                "milestones": timeline.get('trl_4_6', {}).get('milestones', [
                    "Technology validated in lab",
                    "Prototype development",
                    "Technology demonstrated in relevant environment"
                ])
            },
            {
                "name": "Pilot & Demonstration",
                "trls": [7, 8],
                "timeline": timeline.get('trl_7_9', {}).get('timeframe', '12-18 months'),
                "milestones": timeline.get('trl_7_9', {}).get('milestones', [
                    "System prototype demonstration",
                    "System complete and qualified"
                ])
            },
            {
                "name": "Market Deployment",
                "trls": [9],
                "timeline": "18+ months",
                "milestones": [
                    "Actual system proven in operational environment",
                    "Commercial scaling",
                    "Market penetration"
                ]
            }
        ]
        
        # Find current phase index
        current_phase_index = 0
        for i, phase in enumerate(phases):
            if current_trl in phase['trls']:
                current_phase_index = i
                break
        
        return {
            "phases": phases,
            "current_trl": current_trl,
            "current_phase_index": current_phase_index,
            "current_phase": phases[current_phase_index] if current_phase_index < len(phases) else phases[0],
            "time_to_market": timeline.get('time_to_market', '18-24 months')
        }
    
    # NEW METHOD: Generate Highlights & Lowlights
    def _write_highlights_lowlights(self, detailed_viability_assessment: Dict) -> Dict[str, Any]:
        """
        Generate top and bottom performing parameters for UI
        Returns sorted lists and cluster averages
        """
        all_params = []
        cluster_scores = {}
        
        # Flatten the nested structure: cluster > subcategory > parameter
        for cluster_name, cluster_data in detailed_viability_assessment.items():
            cluster_total = 0
            cluster_count = 0
            
            for subcategory_name, subcategory_data in cluster_data.items():
                for param_name, param_data in subcategory_data.items():
                    if isinstance(param_data, dict) and 'assignedScore' in param_data:
                        score = param_data['assignedScore']
                        all_params.append({
                            "name": param_name,
                            "score": score,
                            "clusterName": cluster_name,
                            "paramName": param_name,
                            "subcategory": subcategory_name
                        })
                        cluster_total += score
                        cluster_count += 1
            
            # Calculate average for cluster
            if cluster_count > 0:
                cluster_scores[cluster_name] = round(cluster_total / cluster_count, 1)
        
        # Sort by score
        all_params.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top 3 and bottom 3
        top_performers = all_params[:3]
        bottom_performers = all_params[-3:][::-1]  # Reverse to show lowest first
        
        return {
            "topPerformers": top_performers,
            "bottomPerformers": bottom_performers,
            "allParameters": all_params,
            "avgClusterScores": cluster_scores
        }
    
    # NEW METHOD: Generate Action Points (Next Steps)
    def _write_action_points(self, conclusion: Dict, metadata: Dict, bottom_performers: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate conditional action points based on validation outcome
        Returns UI-ready structure with title and todos array
        """
        outcome = metadata.get('validation_outcome', 'Moderate')
        score = metadata.get('overall_score', 0)
        
        # Determine base action point by outcome
        if outcome in ['Slay', 'Approved'] or score >= 75:
            title = "Accelerate Your Idea"
            todos = [
                "Finalize your business plan and financial projections.",
                "Begin networking with potential investors and partners.",
                "To bring your vision to life, explore professional full-stack development and mobile app services on platforms like Edifai, which covers both tech and non-tech areas."
            ]
        elif outcome in ['Mid', 'Moderate'] or score >= 50:
            title = "Refine and Strengthen"
            todos = [
                "Focus on the 'Areas for Improvement' identified in the report.",
                "Consider a pivot based on the AI's feedback on market fit.",
                "To improve your concept's potential, consider engaging with expert UI/UX design and prototyping services through resources like Edifai, which covers both tech and non-tech areas."
            ]
        else:  # Flop, Rejected, or score < 50
            title = "Rethink and Re-strategize"
            todos = [
                "Deeply analyze the feedback on 'Core Idea & Innovation' and 'Market Need'.",
                "Conduct primary market research to validate the core problem.",
                "For a foundational reset, expert industry consultancy and audit services from a platform like Edifai can provide a new perspective on tech and non-tech topics."
            ]
        
        action_points = [
            {
                "title": title,
                "todos": todos
            }
        ]
        
        # Add specific action point for worst performing parameter if exists
        if bottom_performers:
            worst = bottom_performers[0]  # Lowest score
            action_points.append({
                "title": f"Address Lowest Score: {worst['name']}",
                "todos": [
                    f"Review the feedback for '{worst['name']}' in the detailed assessment.",
                    "Brainstorm 3-5 ways to directly improve this aspect.",
                    "Update your pitch deck to reflect these improvements.",
                    f"Set a 30-day goal to increase {worst['name']} score from {worst['score']} to target of 60+"
                ]
            })
        
        return action_points
    
    # Support methods (existing)
    def _write_explicit_risk_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write comprehensive Risk Analysis section with categorized risks"""
        critical_issues = []
        high_priority_issues = []
        all_scores = [conv.get('score', 0) for conv in conversations]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        for conv in conversations:
            score = conv.get('score', 0)
            area = conv.get('sub_parameter', 'Unknown')
            cluster = conv.get('cluster', 'Unknown')

            if score < 40:
                critical_issues.append({
                    'area': area,
                    'cluster': cluster,
                    'score': score,
                    'weaknesses': conv.get('weaknesses', []),
                    'risks': conv.get('risk_factors', [])
                })
            elif score < 60:
                high_priority_issues.append({
                    'area': area,
                    'cluster': cluster,
                    'score': score,
                    'weaknesses': conv.get('weaknesses', []),
                    'risks': conv.get('risk_factors', [])
                })

        prompt = f"""You are a risk analyst writing a comprehensive Risk Analysis for "{metadata['title']}".

**Critical Issues** (Score < 40/100):
{self._format_issues_for_prompt(critical_issues)}

**High Priority Issues** (Score 40-60/100):
{self._format_issues_for_prompt(high_priority_issues)}

**Overall Score**: {avg_score:.1f}/100

**Write a comprehensive risk analysis with specific risks categorized as**:

1. **Technical Risks** (Technology, development, implementation challenges):
   - Identify 4-5 technical risks from critical/high-priority issues
   - Each risk: description, severity (CRITICAL/HIGH/MEDIUM/LOW), mitigation strategy, likelihood (%), potential impact

2. **Market Risks** (Market adoption, competition, demand challenges):
   - Identify 3-4 market risks
   - Each: description, severity, mitigation, likelihood, impact

3. **Operational Risks** (Execution, team, resource, scalability challenges):
   - Identify 3-4 operational risks
   - Each: description, severity, mitigation, likelihood, impact

4. **Regulatory Risks** (Compliance, legal, regulatory challenges):
   - Identify 2-3 regulatory risks
   - Each: description, severity, mitigation, likelihood, impact

**Important Guidelines**:
- Base ONLY on actual scores and expert feedback provided
- Be specific about each risk
- Severity: CRITICAL (score<40), HIGH (40-60), MEDIUM (60-75), LOW (75+)
- Likelihood: 0-100% based on expert assessment
- Mitigation: Actionable strategies only

Return ONLY JSON:
{{
  "technicalRisks": [
    {{
      "riskDescription": "Specific technical risk",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "mitigation": "Actionable strategy",
      "likelihood": 75,
      "potentialImpact": "Impact description"
    }}
  ],
  "marketRisks": [
    {{"riskDescription": "...", "severity": "...", "mitigation": "...", "likelihood": 60, "potentialImpact": "..."}}
  ],
  "operationalRisks": [
    {{"riskDescription": "...", "severity": "...", "mitigation": "...", "likelihood": 50, "potentialImpact": "..."}}
  ],
  "regulatoryRisks": [
    {{"riskDescription": "...", "severity": "MEDIUM", "mitigation": "...", "likelihood": 30, "potentialImpact": "..."}}
  ],
  "riskScore": 45
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            analysis = json.loads(content)

            # Ensure all risk types present
            analysis.setdefault('technicalRisks', [])
            analysis.setdefault('marketRisks', [])
            analysis.setdefault('operationalRisks', [])
            analysis.setdefault('regulatoryRisks', [])
            analysis.setdefault('riskScore', 100 - int(avg_score))

            logger.info("✅ Risk Analysis completed")
            return analysis

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing risk analysis: {e}")
            return self._create_fallback_risk_analysis(critical_issues, high_priority_issues, avg_score)
    
    def _write_benchmarking_analysis(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Write Benchmarking section with percentile comparison"""
        overall_score = metadata.get('overall_score', 70)
        all_scores = [conv.get('score', 0) for conv in conversations]
        avg_agent_score = sum(all_scores) / len(all_scores) if all_scores else 70

        prompt = f"""You are a benchmarking analyst comparing "{metadata['title']}" to similar startup ideas.

**This Idea Score**: {overall_score:.1f}/100
**Average Expert Score**: {avg_agent_score:.1f}/100

**Your Task**: Write benchmarking insights and establish reference points.

**Benchmarking Framework**:
- Industry Average for startups: 65/100
- Top Quartile (75th percentile): 80/100
- This idea percentile: {int((overall_score/100)*100)}th

**Generate**:

1. **Similar Ideas** (2-3 reference ideas with similar scores):
   - Idea 1 name, score slightly below this idea
   - Idea 2 name, score at this idea's level
   - Idea 3 name, score above this idea
   - Include category and outcome (Approved/Slay/Moderate)

2. **Percentile Position**: Calculate where this idea ranks (0-100%)

3. **Comparison Insights** (4-5 bullet points):
   - How this idea compares to industry average
   - Relative strengths vs comparable ideas
   - Relative weaknesses vs comparable ideas
   - Market positioning
   - Competitive standing

Return ONLY JSON:
{{
  "similarIdeas": [
    {{
      "ideaName": "Name of similar idea",
      "score": 70,
      "category": "Category/Industry",
      "outcome": "Approved|Slay|Moderate|Rejected"
    }}
  ],
  "industryAverage": 65,
  "topQuartile": 80,
  "percentile": {int((overall_score/100)*100)},
  "insights": [
    "insight 1",
    "insight 2",
    "insight 3",
    "insight 4"
  ]
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            analysis = json.loads(content)

            # Ensure required fields
            if not analysis.get('similarIdeas'):
                analysis['similarIdeas'] = [
                    {"ideaName": "Reference Startup A", "score": overall_score-5, "category": "Similar", "outcome": "Approved"},
                    {"ideaName": "Reference Startup B", "score": overall_score+3, "category": "Peer", "outcome": "Slay"}
                ]

            analysis.setdefault('industryAverage', 65)
            analysis.setdefault('topQuartile', 80)
            analysis['percentile'] = int((overall_score/100)*100) if overall_score > 0 else 50
            analysis.setdefault('insights', [
                f"This idea scores {overall_score:.0f}/100 overall.",
                f"Performance is {'above' if overall_score > 65 else 'below'} industry average of 65/100.",
                f"Percentile rank: {analysis['percentile']}th ({('top' if analysis['percentile'] > 75 else 'middle' if analysis['percentile'] > 50 else 'lower')} performance tier)."
            ])

            logger.info("✅ Benchmarking Analysis completed")
            return analysis

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing benchmarking: {e}")
            percentile = int((overall_score/100)*100) if overall_score > 0 else 50
            return {
                "similarIdeas": [
                    {"ideaName": "Reference Startup A", "score": overall_score-5, "category": "Similar", "outcome": "Approved"},
                    {"ideaName": "Reference Startup B", "score": overall_score+3, "category": "Peer", "outcome": "Slay"}
                ],
                "industryAverage": 65,
                "topQuartile": 80,
                "percentile": percentile,
                "insights": [
                    f"Score: {overall_score:.0f}/100",
                    f"Performance vs average: {'Above' if overall_score > 65 else 'Below'}",
                    f"Percentile: {percentile}th"
                ]
            }
    
    def _write_structured_recommendations(self, conclusion: Dict, market_analysis: Dict, trl_analysis: Dict, metadata: Dict) -> Dict[str, Any]:
        """Write structured Recommendations section with phases and resources"""
        path_forward = conclusion.get('path_forward', [])
        final_verdict = conclusion.get('final_verdict', [])
        trl_level = trl_analysis.get('current_trl', {}).get('level', 3) if trl_analysis else 3

        prompt = f"""You are a strategic advisor writing Structured Recommendations for "{metadata['title']}".

**Context**:
- Score: {metadata.get('overall_score', 70)}/100
- Path Forward (from conclusion): {path_forward[:3] if path_forward else ['None']}
- Current TRL Level: {trl_level}/9
- Market: {market_analysis.get('opportunity_summary', ['Large market'])[:1] if market_analysis else ['Market exists']}

**Write structured recommendations with**:

1. **Immediate Actions** (Next 0-3 months):
   - 5-6 specific, actionable items
   - Each with: priority (CRITICAL/HIGH/MEDIUM), estimated effort (weeks/months), expected outcome

2. **Long-Term Strategy** (3 phases, 12-18 months):
   - Phase 1 (Months 1-3): Foundation, MVP, initial validation
   - Phase 2 (Months 4-9): Growth, scaling, market expansion
   - Phase 3 (Months 10-18+): Scale, profitability, sustainability
   - Each phase: objectives, dependencies, timeline, success metrics

3. **Resources Required**:
   - skillsNeeded (list top 5 critical skills)
   - budgetEstimate (in INR, reasonable estimate)
   - timelineMonths (total project timeline)
   - partnershipOpportunities (strategic partners, institutions, government programs)

**Guidelines**:
- Be specific and actionable
- Base on actual score and insights
- Focus on moving from current state to success
- Include realistic timelines
- Prioritize by impact

Return ONLY JSON:
{{
  "immediateActions": [
    {{
      "priority": "CRITICAL|HIGH|MEDIUM",
      "action": "Specific action",
      "rationale": "Why this matters",
      "estimatedEffort": "2 weeks|1 month|3 months",
      "expectedOutcome": "Specific outcome"
    }}
  ],
  "longTermStrategy": [
    {{
      "phase": "Phase 1: Foundation",
      "objectives": ["Objective 1", "Objective 2"],
      "dependencies": ["Dependency 1", "Dependency 2"],
      "timelineMonths": 3,
      "successMetrics": ["Metric 1", "Metric 2"]
    }}
  ],
  "resources": {{
    "skillsNeeded": ["Skill 1", "Skill 2", "Skill 3"],
    "budgetEstimate": 500000,
    "timelineMonths": 18,
    "partnershipOpportunities": ["Partner 1", "Partner 2"]
  }}
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            recommendations = json.loads(content)

            # Ensure all sections present
            if not recommendations.get('immediateActions'):
                recommendations['immediateActions'] = self._generate_default_immediate_actions(path_forward)

            if not recommendations.get('longTermStrategy'):
                recommendations['longTermStrategy'] = self._generate_default_strategy()

            recommendations.setdefault('resources', self._generate_default_resources())

            logger.info("✅ Structured Recommendations completed")
            return recommendations

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error writing recommendations: {e}")
            return {
                "immediateActions": self._generate_default_immediate_actions(path_forward),
                "longTermStrategy": self._generate_default_strategy(),
                "resources": self._generate_default_resources()
            }
    
    def _write_detailed_viability_assessment(self, conversations: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """
        Write Detailed Viability Assessment with hierarchical structure:
        Cluster > Sub-Category > Parameter (with score, whatWentWell, whatCanBeImproved)
        """
        self._update_progress("Generating Detailed Viability Assessment...", 90)
        logger.info("📊 Generating Detailed Viability Assessment...")
        
        viability_assessment = {}
        
        # Group conversations by cluster
        cluster_groups = {}
        for conv in conversations:
            cluster = conv.get('cluster', 'Unknown')
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append(conv)
            
        for cluster_name, cluster_convs in cluster_groups.items():
            logger.info(f"Processing cluster: {cluster_name}")
            
            # Further group by parameter (sub-category)
            parameter_groups = {}
            for conv in cluster_convs:
                param = conv.get('parameter', 'General')
                if param not in parameter_groups:
                    parameter_groups[param] = []
                parameter_groups[param].append(conv)
                
            cluster_data = {}
            
            for parameter_name, parameter_convs in parameter_groups.items():
                parameter_data = {}
                
                for conv in parameter_convs:
                    sub_parameter_name = conv.get('sub_parameter', 'Unknown Parameter')
                    score = conv.get('score', 0)
                    explanation = conv.get('explanation', '')
                    strengths = conv.get('strengths', [])
                    weaknesses = conv.get('weaknesses', [])
                    
                    # Extract "What Went Well" from strengths
                    what_went_well = '; '.join(strengths[:2]) if strengths else explanation[:150] + "..." if explanation else "Performance meets expectations"
                    
                    # Extract "What Can Be Improved" from weaknesses
                    what_can_be_improved = '; '.join(weaknesses[:2]) if weaknesses else "Continue monitoring for optimization opportunities"
                    
                    parameter_data[sub_parameter_name] = {
                        "assignedScore": round(score, 1),
                        "whatWentWell": what_went_well,
                        "whatCanBeImproved": what_can_be_improved
                    }
                
                cluster_data[parameter_name] = parameter_data
                
            viability_assessment[cluster_name] = cluster_data
            
        logger.info(f"✅ Detailed Viability Assessment completed with {len(viability_assessment)} clusters")
        return viability_assessment
    
    # Fallback methods (existing)
    def _create_fallback_cluster_report(self, cluster_name: str, conversations: List[Dict]) -> Dict:
        """Create fallback report if AI writing fails"""
        parameters = []
        for conv in conversations:
            parameters.append({
                'name': conv.get('sub_parameter', 'Unknown'),
                'score': conv.get('score', 0),
                'findings': [conv.get('explanation', '')],
                'strengths': conv.get('strengths', []),
                'weaknesses': conv.get('weaknesses', []),
                'recommendations': conv.get('recommendations', [])
            })
        
        scores = [p['score'] for p in parameters]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'cluster_name': cluster_name,
            'cluster_score': avg_score,
            'overview': [f"Analysis based on {len(parameters)} expert evaluations"],
            'parameters': parameters,
            'cluster_summary': [f"Average score: {avg_score:.1f}/100"]
        }
    
    def _create_fallback_summary(self, strengths: List, weaknesses: List, avg_score: float, metadata: Dict) -> Dict:
        """Create fallback summary"""
        return {
            'overall_score': avg_score,
            'outcome': metadata['validation_outcome'],
            'key_findings': [f"Overall validation score: {avg_score:.1f}/100"],
            'major_strengths': [s['text'] for s in strengths[:5]],
            'critical_concerns': [w['text'] for w in weaknesses[:5]],
            'strategic_recommendations': ["Review detailed cluster analysis for specific actions"]
        }
    
    def _create_fallback_conclusion(self, score: float, metadata: Dict) -> Dict:
        """Create fallback conclusion"""
        return {
            'investment_decision': self._get_investment_decision(score),
            'final_verdict': [f"Validation score: {score:.1f}/100"],
            'path_forward': ["Review detailed recommendations in each category"],
            'success_factors': ["Address identified weaknesses", "Leverage existing strengths"],
            'risk_mitigation': ["Monitor areas scoring below 50/100"],
            'market_assessment': ["Review market analysis section for detailed TAM/SAM/SOM"]
        }
    
    def _create_fallback_market_analysis(self, metadata: Dict) -> Dict:
        """Create fallback market analysis"""
        return {
            'tam': {
                'definition': 'Total Addressable Market - Global market size',
                'size': 'To be determined based on market research',
                'growth_rate': 'N/A',
                'assumptions': ['Market size requires detailed research', 'Growth rate depends on industry trends'],
                'trends': ['Market trends to be analyzed']
            },
            'sam': {
                'definition': 'Serviceable Available Market - Addressable market segment',
                'size': 'To be determined',
                'geographic_focus': 'Primary markets to be identified',
                'accessibility_factors': ['Market accessibility requires analysis'],
                'demographics': ['Target demographics to be defined']
            },
            'som': {
                'definition': 'Serviceable Obtainable Market - Realistic market share',
                'size': 'To be determined',
                'market_share': 'Target market share to be calculated',
                'competitive_landscape': ['Competitive analysis required'],
                'capture_strategy': ['Market capture strategy to be developed']
            },
            'opportunity_summary': ['Market analysis requires detailed research and validation']
        }
    
    def _create_fallback_trl_analysis(self, metadata: Dict) -> Dict:
        """Create fallback TRL analysis"""
        return {
            'current_trl': {
                'level': 3,
                'justification': ['Technology readiness requires technical assessment'],
                'components_status': ['Component status to be evaluated']
            },
            'timeline': {
                'trl_1_3': {
                    'timeframe': 'To be determined',
                    'milestones': ['Initial milestones to be defined']
                },
                'trl_4_6': {
                    'timeframe': 'To be determined',
                    'milestones': ['Development milestones to be planned']
                },
                'trl_7_9': {
                    'timeframe': 'To be determined',
                    'milestones': ['Market readiness milestones to be established']
                },
                'time_to_market': 'To be determined based on TRL progression'
            },
            'risks': ['Technology risks require detailed assessment'],
            'strengths': ['Technology strengths to be identified'],
            'recommendations': ['TRL advancement recommendations require technical review']
        }
    
    def _create_fallback_pros_cons(self, pros: List[Dict], cons: List[Dict]) -> Dict:
        """Create fallback pros/cons analysis"""
        return {
            'advantages': [p['text'] for p in pros[:8]] if pros else ['Advantages to be identified'],
            'disadvantages': [c['text'] for c in cons[:8]] if cons else ['Disadvantages to be identified'],
            'balanced_assessment': ['Balanced assessment requires comprehensive review']
        }
    
    def _create_fallback_weaknesses(self, weaknesses: List[Dict]) -> Dict:
        """Create fallback weaknesses analysis"""
        critical = [w for w in weaknesses if w['severity'] == 'Critical']
        high = [w for w in weaknesses if w['severity'] == 'High']
        moderate = [w for w in weaknesses if w['severity'] == 'Moderate']
        
        return {
            'critical': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Immediate attention required'}
                for w in critical[:6]
            ] if critical else [],
            'high_priority': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Short-term improvement needed'}
                for w in high[:7]
            ] if high else [],
            'moderate': [
                {'weakness': w['text'], 'impact': f"Score: {w['score']:.1f}/100", 'action': 'Medium-term improvement plan'}
                for w in moderate[:6]
            ] if moderate else [],
            'patterns': ['Weakness patterns require detailed analysis'],
            'remediation_strategy': ['Remediation strategy to be developed based on identified weaknesses']
        }
    
    def _create_fallback_risk_analysis(self, critical_issues, high_priority_issues, avg_score):
        """Create fallback risk analysis"""
        tech_risks = []
        market_risks = []
        operational_risks = []

        for issue in critical_issues[:2]:
            tech_risks.append({
                "riskDescription": f"{issue['area']} - {issue['cluster']} (Score: {issue['score']:.0f})",
                "severity": "CRITICAL",
                "mitigation": "Immediate action required",
                "likelihood": 85,
                "potentialImpact": "Project blocking"
            })

        for issue in high_priority_issues[:2]:
            market_risks.append({
                "riskDescription": f"{issue['area']} in {issue['cluster']} domain",
                "severity": "HIGH",
                "mitigation": "Develop mitigation strategy",
                "likelihood": 65,
                "potentialImpact": "Market impact"
            })

        operational_risks.append({
            "riskDescription": "Execution and resource constraints",
            "severity": "MEDIUM",
            "mitigation": "Build operational capacity",
            "likelihood": 50,
            "potentialImpact": "Timeline delays"
        })

        return {
            "technicalRisks": tech_risks or [{"riskDescription": "Technology validation needed", "severity": "HIGH", "mitigation": "Conduct technical review", "likelihood": 60, "potentialImpact": "Development delays"}],
            "marketRisks": market_risks or [{"riskDescription": "Market adoption risk", "severity": "MEDIUM", "mitigation": "Market research", "likelihood": 55, "potentialImpact": "Adoption"}],
            "operationalRisks": operational_risks,
            "regulatoryRisks": [{"riskDescription": "Regulatory changes", "severity": "LOW", "mitigation": "Monitor landscape", "likelihood": 30, "potentialImpact": "Compliance costs"}],
            "riskScore": 100 - int(avg_score)
        }
    
