"""
Base Agent Class for CrewAI Validation System
Provides common functionality for all validation agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from crewai import Agent
from langchain_openai import ChatOpenAI
from datetime import datetime
import json


class BaseValidationAgent(ABC):
    """
    Base class for all validation agents in the CrewAI system.
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, agent_id: str, cluster: str, parameter: str, 
                 sub_parameter: str, weight: float, dependencies: List[str], llm: ChatOpenAI):
        self.agent_id = agent_id
        self.cluster = cluster
        self.parameter = parameter
        self.sub_parameter = sub_parameter
        self.weight = weight
        self.dependencies = dependencies
        self.llm = llm
        self._agent = None
        self._last_evaluation = None
    
    def create_agent(self) -> Agent:
        """Create the CrewAI agent instance. Override in subclasses for customization."""
        if self._agent is None:
            self._agent = Agent(
                role=f"{self.sub_parameter} Validation Specialist",
                goal=self.get_default_goal(),
                backstory=self.get_default_backstory(),
                verbose=False,  # Disable verbose for speed
                allow_delegation=False,  # Disable delegation for speed
                llm=self.llm,
                tools=self.get_specialized_tools(),
                max_iter=1,  # Single iteration for speed
                memory=False  # Disable memory for speed
            )
        return self._agent
    
    def get_default_goal(self) -> str:
        """Default goal for the agent"""
        return f"""Provide expert evaluation of {self.sub_parameter} for startup ideas with precise 
        scoring (0-100 scale), concise bullet-point analysis, and actionable insights within the {self.cluster} 
        evaluation framework. Be CRITICAL and RIGOROUS in your assessment. Challenge assumptions, 
        identify weaknesses, and don't hesitate to give low scores for poor ideas. Question other 
        agents' evaluations when they seem too lenient. Focus on Indian market context and provide 
        specific, measurable assessments with HONEST scoring. Use bullet points for clarity."""
    
    def get_default_backstory(self) -> str:
        """Default backstory for the agent"""
        backstories = {
            "Core Idea": """You are a seasoned innovation consultant with deep expertise in evaluating 
            breakthrough ideas and disruptive technologies. You have helped assess hundreds of startups 
            and understand what makes ideas truly innovative in the Indian market context. You've seen 
            too many overhyped ideas fail and are known for your RIGOROUS, CRITICAL analysis. You don't 
            give participation trophies - you tell the hard truth about idea viability.""",
            
            "Market Opportunity": """You are a market research expert with extensive experience in the 
            Indian startup ecosystem. You understand market dynamics, customer behavior, and growth 
            potential in emerging markets, with specific expertise in Indian consumer and business segments.""",
            
            "Execution": """You are a technical and operational expert who has guided numerous startups 
            through execution challenges. You understand the complexities of building and scaling 
            technology solutions in the Indian infrastructure and talent landscape.""",
            
            "Business Model": """You are a business strategy expert with deep knowledge of sustainable 
            business models and financial viability. You have experience with venture capital and startup 
            valuations in the Indian market, understanding local business dynamics and investor preferences.""",
            
            "Team": """You are an organizational development expert who understands what makes 
            high-performing teams. You have experience in founder coaching and team building for 
            startups, with deep knowledge of Indian workplace culture and talent dynamics.""",
            
            "Compliance": """You are a regulatory and compliance expert with specialized knowledge of 
            Indian business environment, ESG principles, and ecosystem dynamics. You understand the 
            complex regulatory landscape and sustainability requirements for Indian startups.""",
            
            "Risk & Strategy": """You are a strategic risk assessment expert who helps startups navigate 
            uncertainties and position themselves for investment and growth opportunities. You have 
            deep experience with Indian market risks and strategic positioning."""
        }
        
        base_backstory = backstories.get(self.cluster, 
            "You are a specialized validation expert with deep domain knowledge.")
        
        return f"""{base_backstory} Your specific expertise lies in {self.sub_parameter} evaluation, 
        and you collaborate with other specialists to provide comprehensive assessments. You are known 
        for challenging other experts when their assessments seem too optimistic or lack sufficient 
        evidence. You believe in rigorous evaluation and aren't afraid to disagree with colleagues 
        when the data doesn't support their conclusions."""
    
    def get_specialized_tools(self) -> List:
        """Get specialized tools for this agent. Override in subclasses."""
        return []
    
    def _determine_industry_context(self, idea_name: str, idea_concept: str) -> str:
        """Determine the industry context from idea name and concept"""
        text = f"{idea_name} {idea_concept}".lower()
        
        if any(word in text for word in ['food', 'delivery', 'restaurant', 'meal', 'cooking', 'nutrition', 'lunch', 'breakfast', 'dinner', 'snacks']):
            return "Food & Delivery"
        elif any(word in text for word in ['health', 'medical', 'healthcare', 'doctor', 'patient', 'medicine', 'hospital']):
            return "Healthcare"
        elif any(word in text for word in ['education', 'learning', 'school', 'university', 'course', 'student', 'teach', 'academic']):
            return "Education"
        elif any(word in text for word in ['finance', 'banking', 'payment', 'money', 'investment', 'fintech', 'financial']):
            return "Finance"
        elif any(word in text for word in ['ecommerce', 'retail', 'shopping', 'marketplace', 'store', 'buy', 'sell']):
            return "E-commerce"
        elif any(word in text for word in ['manufacturing', 'production', 'factory', 'industrial', 'manufacture']):
            return "Manufacturing"
        elif any(word in text for word in ['tech', 'software', 'app', 'platform', 'digital', 'ai', 'technology']):
            return "Technology"
        else:
            return "General Business"
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get evaluation criteria specific to this agent. Override in subclasses."""
        return {
            "default_criteria": {
                "description": f"Evaluate {self.sub_parameter} thoroughly",
                "scoring_rubric": {
                    "90-100": "Outstanding - exceeds expectations significantly",
                    "80-89": "Strong - above market standards",
                    "70-79": "Good - meets expectations well",
                    "60-69": "Acceptable - meets basic requirements",
                    "50-59": "Below expectations - needs improvement",
                    "40-49": "Weak - major improvements required",
                    "30-39": "Poor - fundamental problems",
                    "0-29": "Critical - not viable"
                }
            }
        }
    
    def get_collaboration_dependencies(self) -> Dict[str, str]:
        """Get information about collaboration with other agents. Override in subclasses."""
        return {}
    
    def create_evaluation_prompt(self, idea_name: str, idea_concept: str, 
                               dependency_results: Optional[Dict[str, Any]] = None) -> str:
        """Create a comprehensive evaluation prompt for the agent"""
        
        # Determine industry context
        industry_context = self._determine_industry_context(idea_name, idea_concept)
        
        criteria = self.get_evaluation_criteria()
        
        prompt = f"""
        VALIDATION TASK: {self.sub_parameter} Assessment

        IDEA DETAILS:
        - Name: {idea_name}
        - Concept: {idea_concept}
        - Industry Context: {industry_context}

        YOUR SPECIALIZATION:
        You are evaluating {self.sub_parameter} within the {self.cluster} cluster.
        Weight in overall assessment: {self.weight}

        # ... keep all the existing sections ...

        REQUIRED OUTPUT FORMAT (BULLET POINTS ONLY):
        Provide a JSON response with exactly this structure:
        - score: integer or float between 0 and 100
        - confidence_level: float between 0.0 and 1.0
        - explanation: Detailed analysis (maximum 75 words, be comprehensive but concise)
        - key_insights: [Bullet point insight 1 - one line max, Bullet point insight 2 - one line max, Bullet point insight 3 - one line max]
        - strengths: [Strength 1 - one line, Strength 2 - one line]
        - weaknesses: [Weakness 1 - one line - BE SPECIFIC, Weakness 2 - one line - BE SPECIFIC, Weakness 3 - one line - BE SPECIFIC]
        - recommendations: [Action item 1 - one line, Action item 2 - one line, Action item 3 - one line]
          MANDATORY: Always provide at least 2 recommendations, even for high scores
        - risk_factors: [Risk 1 - one line, Risk 2 - one line]
        - assumptions: [Assumption 1, Assumption 2]
        - peer_challenges: [Challenge 1, Challenge 2]
        - evidence_gaps: [Gap 1, Gap 2]
        - indian_market_considerations: ONE sentence about Indian market

        IMPORTANT CONSTRAINTS (MANDATORY - VIOLATION WILL FAIL VALIDATION):
        1. explanation: MUST be 25-75 words (NOT empty, NOT one sentence)
        2. recommendations: MUST contain EXACTLY 3 actionable items (NO EMPTY ARRAYS)
        3. weaknesses: MUST contain AT LEAST 2 specific weaknesses
        4. strengths: MUST contain AT LEAST 2 specific strengths
        5. key_insights: MUST contain 2-3 key insights
        6. ALL fields must be populated with specific, actionable content

        CRITICAL: Your response will be rejected if any field is empty or contains generic placeholder text.
        """

        
        return prompt
    
    def execute(self, context: Dict) -> Dict:
        """Execute agent and enforce CRITICAL data population"""
        agent = self.create_agent()
        
        prompt = self.create_evaluation_prompt(
            context.get('idea_name', ''),
            context.get('idea_concept', ''),
            context.get('dependency_results')
        )
        
        # Run agent
        raw_output = agent.run(prompt)
        
        # Validate output
        result = self.validate_output(raw_output)
        
        # ✅ CRITICAL: Force recommendations (what_can_be_improved)
        if not result['recommendations'] or len(result['recommendations']) < 3:
            logger.warning(f"⚠️ Forcing recommendations for {self.sub_parameter}")
            result['recommendations'] = [
                f"Validate {self.sub_parameter} assumptions through market research",
                f"Benchmark {self.sub_parameter} against top 3 competitors",
                f"Develop improvement roadmap for {self.sub_parameter} based on findings"
            ]
        
        # Force weaknesses
        if not result['weaknesses'] or len(result['weaknesses']) < 2:
            result['weaknesses'] = [
                f"Limited validation data for {self.sub_parameter}",
                f"Potential scalability challenges in {self.sub_parameter}"
            ]
        
        # Force strengths
        if not result['strengths'] or len(result['strengths']) < 2:
            result['strengths'] = [
                f"Foundational framework for {self.sub_parameter} established",
                f"Initial assessment of {self.sub_parameter} feasibility completed"
            ]
        
        # Ensure explanation is 50 words max
        words = result['explanation'].split()
        if len(words) > 50:
            result['explanation'] = ' '.join(words[:50]) + '...'
        
        return result

    
    
    def _format_criteria(self, criteria: Dict[str, Any]) -> str:
        """Format evaluation criteria for the prompt"""
        formatted = ""
        for criterion_name, criterion_data in criteria.items():
            formatted += f"\n{criterion_name.upper()}:\n"
            formatted += f"- {criterion_data.get('description', 'No description')}\n"
            
            if 'scoring_rubric' in criterion_data:
                formatted += "Scoring Rubric:\n"
                for score, desc in criterion_data['scoring_rubric'].items():
                    formatted += f"  {score}: {desc}\n"
            
            if 'factors' in criterion_data:
                formatted += f"Key Factors: {', '.join(criterion_data['factors'])}\n"
        
        return formatted
    
    def _format_dependency_results(self, dependency_results: Optional[Dict[str, Any]]) -> str:
        """Format dependency results for the prompt"""
        if not dependency_results:
            return "No dependency insights available for this evaluation."
        
        formatted = "🎯 PREVIOUS AGENT ASSESSMENTS TO ANALYZE & CHALLENGE:\n"
        formatted += "Your job is to scrutinize these assessments and provide counterarguments.\n\n"
        
        for agent_name, result in dependency_results.items():
            score = result.get('assigned_score', result.get('score', 'N/A'))
            explanation = result.get('explanation', 'No explanation available')
            confidence = result.get('confidence_level', 'N/A')
            assumptions = result.get('assumptions', [])
            
            formatted += f"**{agent_name} Assessment:**\n"
            formatted += f"- Score: {score}/5.0 (Confidence: {confidence})\n"
            formatted += f"- Reasoning: {explanation}\n"
            if assumptions:
                formatted += f"- Their Assumptions: {', '.join(assumptions)}\n"
            
            # Add challenge prompts based on score
            if isinstance(score, (int, float)):
                if score > 3.5:
                    formatted += f"⚠️ CHALLENGE THIS: Score seems too optimistic. Find flaws in their reasoning.\n"
                elif score < 2.5:
                    formatted += f"⚠️ CHALLENGE THIS: Score might be too harsh. Consider if they missed positives.\n"
                else:
                    formatted += f"⚠️ CHALLENGE THIS: Question their methodology and evidence.\n"
            formatted += "\n"
        
        return formatted
    
    def validate_output(self, output: Any) -> Dict[str, Any]:
        """Validate and standardize agent output - ENSURES NO EMPTY FIELDS"""
        try:
            if isinstance(output, str):
                try:
                    # Try JSON parsing first
                    result = json.loads(output)
                except:
                    # Fallback to text parsing
                    result = self._parse_text_output(output)
            else:
                result = output

            # ENSURE ALL fields exist and are not empty
            standardized = {
                "score": float(result.get("score", 70.0)),
                "confidence_level": float(result.get("confidence_level", 0.7)),
                "explanation": str(result.get("explanation", f"Evaluation of {self.sub_parameter}")),
                "assumptions": list(result.get("assumptions", ["Standard market assumptions"])),
                "key_insights": list(result.get("key_insights", [])),
                "recommendations": list(result.get("recommendations", [])),
                "risk_factors": list(result.get("risk_factors", [])),
                "strengths": list(result.get("strengths", [])),
                "weaknesses": list(result.get("weaknesses", [])),
            }

            # ✅ CRITICAL: NEVER return empty arrays
            if not standardized['recommendations']:
                standardized['recommendations'] = ["Validation required", "Market research needed", "Competitive analysis recommended"]

            if not standardized['weaknesses']:
                standardized['weaknesses'] = ["Needs validation", "Scalability unclear"]

            if not standardized['strengths']:
                standardized['strengths'] = ["Basic framework present", "Initial concept valid"]

            if not standardized['key_insights']:
                standardized['key_insights'] = ["Requires deeper analysis", "Market context critical"]

            return standardized

        except Exception as e:
            logger.error(f"Validation failed for {self.agent_id}: {e}")
            return self._create_fallback_result()


    def _parse_text_output(self, text: str) -> Dict[str, Any]:
        """Parse text output when JSON parsing fails"""
        import re
        
        # Try to extract score
        score_match = re.search(r'score[:\s]*([0-9\.]+)', text.lower())
        score = float(score_match.group(1)) if score_match else 3.0
        
        # Extract explanation (first substantial paragraph)
        sentences = text.split('.')[:3]
        explanation = '. '.join(sentences).strip()
        
        return {
            "score": score,
            "confidence_level": 0.6,
            "explanation": explanation or f"Text-based evaluation for {self.sub_parameter}",
            "assumptions": ["Extracted from text analysis"],
            "key_insights": [],
            "recommendations": [],
            "risk_factors": [],
            "indian_market_considerations": "Standard considerations applied"
        }
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when validation fails"""
        return {
            "score": 3.0,
            "confidence_level": 0.5,
            "explanation": f"Fallback evaluation for {self.sub_parameter} due to processing error",
            "assumptions": ["Fallback evaluation applied"],
            "key_insights": ["Requires manual review"],
            "recommendations": ["Review evaluation methodology"],
            "risk_factors": ["Evaluation uncertainty"],
            "indian_market_considerations": "Standard market factors assumed"
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this agent"""
        return {
            "agent_id": self.agent_id,
            "cluster": self.cluster,
            "parameter": self.parameter,
            "sub_parameter": self.sub_parameter,
            "weight": self.weight,
            "dependencies": self.dependencies,
            "evaluation_criteria": self.get_evaluation_criteria(),
            "collaboration_info": self.get_collaboration_dependencies()
        }
    
    def record_evaluation(self, result: Dict[str, Any], processing_time: float):
        """Record the last evaluation for debugging and analysis"""
        self._last_evaluation = {
            "result": result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_last_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get the last evaluation performed by this agent"""
        return self._last_evaluation


class AgentCollaborationManager:
    """Manages collaboration and dependencies between agents"""
    
    def __init__(self, agents: Dict[str, BaseValidationAgent]):
        self.agents = agents
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a dependency graph for agent execution order"""
        graph = {}
        for agent_id, agent in self.agents.items():
            graph[agent_id] = agent.dependencies
        return graph
    
    def get_execution_order(self) -> List[List[str]]:
        """Get execution order respecting dependencies (topological sort)"""
        # Simple dependency resolution - can be enhanced
        independent = []
        dependent = []
        
        for agent_id, agent in self.agents.items():
            if not agent.dependencies:
                independent.append(agent_id)
            else:
                dependent.append(agent_id)
        
        return [independent, dependent]  # Simplified for now
    
    def resolve_dependencies(self, agent_id: str, completed_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dependencies for a specific agent"""
        agent = self.agents[agent_id]
        dependency_results = {}
        
        for dependency in agent.dependencies:
            # Find which agent evaluated this dependency
            for eval_agent_id, result in completed_evaluations.items():
                eval_agent = self.agents.get(eval_agent_id)
                if eval_agent and dependency.lower().replace(' ', '_') in eval_agent.sub_parameter.lower().replace(' ', '_'):
                    dependency_results[dependency] = result
                    break
        
        return dependency_results
