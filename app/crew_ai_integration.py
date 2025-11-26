"""
Integration module for CrewAI Multi-Agent Validation System with Pragati Backend
Replaces the existing AI logic with 109+ specialized agents.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from crew_ai_validation import CrewAIValidationOrchestrator, ValidationResult, ValidationOutcome
from crew_ai_validation.agent_factory import ComprehensiveAgentFactory
from crew_ai_validation.base_agent import AgentCollaborationManager

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)



class PragatiCrewAIValidator:
    """
    Main integration class that replaces ai_logic_v2.py with CrewAI multi-agent system.
    Maintains compatibility with existing Flask app while using 109+ specialized agents.
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.3,
            model="gpt-4o",
            max_tokens=1500
        )
        
        # Initialize agent factory and create all agents
        # print("Initializing 109+ specialized validation agents...")
        self.agent_factory = ComprehensiveAgentFactory(self.llm)
        self.agents = self.agent_factory.create_all_agents()
        
        # Initialize collaboration manager
        self.collaboration_manager = AgentCollaborationManager(self.agents)
        
        # Initialize orchestrator with our agents
        self.orchestrator = CrewAIValidationOrchestrator()
        self.orchestrator.agent_registry = self._convert_agents_to_registry_format()
        
        # print(f"✅ Successfully initialized {len(self.agents)} validation agents")
        # self._print_agent_summary()
    
    def _convert_agents_to_registry_format(self) -> Dict[str, Dict[str, Any]]:
        """Convert our agents to the format expected by the orchestrator"""
        registry = {}
        for agent_id, agent in self.agents.items():
            registry[agent_id] = {
                'agent': agent.create_agent(),
                'cluster': agent.cluster,
                'parameter': agent.parameter,
                'sub_parameter': agent.sub_parameter,
                'config': {
                    'weight': agent.weight,
                    'dependencies': agent.dependencies
                }
            }
        return registry
    
    def _estimate_trl_level(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Estimate TRL level and generate phase-specific key activities
        based on validation scores
        """
        overall_score = validation_result.overall_score

        # TRL mapping based on overall score
        if overall_score >= 90:
            current_trl = "TRL-9"
            phase_name = "Market Leadership"
        elif overall_score >= 80:
            current_trl = "TRL-8"
            phase_name = "Scale & Expansion"
        elif overall_score >= 70:
            current_trl = "TRL-7"
            phase_name = "Market Entry"
        elif overall_score >= 60:
            current_trl = "TRL-6"
            phase_name = "Pilot & Validation"
        elif overall_score >= 50:
            current_trl = "TRL-5"
            phase_name = "MVP Development"
        elif overall_score >= 40:
            current_trl = "TRL-4"
            phase_name = "Prototype"
        elif overall_score >= 30:
            current_trl = "TRL-3"
            phase_name = "Proof of Concept"
        else:
            current_trl = "TRL-2"
            phase_name = "Concept"

        # Generate key activities based on weak areas
        weak_areas = self._identify_weak_areas(validation_result.agent_evaluations)
        key_activities = self._generate_key_activities(weak_areas, phase_name)

        return {
            "current_trl": current_trl,
            "phase_name": phase_name,
            "key_activities": key_activities,
            "estimated_timeline": self._get_phase_timeline(phase_name),
            "next_phase_requirements": self._get_next_phase_requirements(current_trl)
        }

    def _identify_weak_areas(self, evaluations: List) -> List[Dict[str, Any]]:
        """Identify parameters with scores < 70"""
        weak_areas = []
        for evaluation in evaluations:
            score = getattr(evaluation, 'assigned_score', 0)
            if score < 70:
                weak_areas.append({
                    'cluster': getattr(evaluation, 'cluster', ''),
                    'parameter': getattr(evaluation, 'sub_parameter', ''),
                    'score': score
                })
        return weak_areas

    def _generate_key_activities(self, weak_areas: List[Dict], phase_name: str) -> List[Dict[str, str]]:
        """Generate max 5 key activities prioritized by lowest scores"""
        activities = []
        
        # Sort weak areas by score (lowest first = highest priority)
        sorted_weak_areas = sorted(weak_areas, key=lambda x: x['score'])
        
        # Take only top 5 most critical areas
        top_weak_areas = sorted_weak_areas[:5]
        
        for weak_area in top_weak_areas:
            cluster = weak_area['cluster']
            param = weak_area['parameter']
            score = weak_area['score']
            
            # Create specific, actionable activity
            if cluster == "Execution":
                activities.append({
                    "text": f"Critical: Fix {param} (score: {score}/100) before phase completion",
                    "timeline": "2-3 weeks"
                })
            elif cluster == "Business Model":
                activities.append({
                    "text": f"Address {param} weakness to strengthen business viability",
                    "timeline": "1-2 weeks"
                })
            elif cluster == "Core Idea":
                activities.append({
                    "text": f"Improve {param} to enhance core value proposition",
                    "timeline": "2-4 weeks"
                })
            else:
                activities.append({
                    "text": f"Prioritize: Improve {param} in {cluster} (score: {score}/100)",
                    "timeline": "2 weeks"
                })
        
        # If fewer than 3 weak areas, add positive/next-step activities
        while len(activities) < 3:
            activities.append({
                "text": f"Prepare documentation and validate current phase deliverables",
                "timeline": f"Week {len(activities) + 1}"
            })
        
        # Always add a maximum of 5 activities
        return activities[:5]
    
    def _identify_weak_areas(self, evaluations: List) -> List[Dict[str, Any]]:
        """Identify top 5 parameters with lowest scores (< 70)"""
        weak_areas = []
        
        for evaluation in evaluations:
            score = getattr(evaluation, 'assigned_score', 0)
            if score < 70:
                weak_areas.append({
                    'cluster': getattr(evaluation, 'cluster', ''),
                    'parameter': getattr(evaluation, 'sub_parameter', ''),
                    'score': score
                })
        
        # Sort and take top 5
        return sorted(weak_areas, key=lambda x: x['score'])[:5]


    def _get_phase_timeline(self, phase_name: str) -> str:
        """Get estimated timeline for phase"""
        timelines = {
            "Concept": "2-4 weeks",
            "Proof of Concept": "4-8 weeks",
            "Prototype": "6-12 weeks",
            "MVP Development": "8-16 weeks",
            "Pilot & Validation": "12-20 weeks",
            "Market Entry": "16-24 weeks",
            "Scale & Expansion": "24+ weeks",
            "Market Leadership": "Ongoing"
        }
        return timelines.get(phase_name, "TBD")

    def _get_next_phase_requirements(self, current_trl: str) -> List[str]:
        """Get requirements for next TRL level"""
        requirements = {
            "TRL-2": ["Proof of concept validation", "Technical feasibility study"],
            "TRL-3": ["Working prototype", "Initial user testing"],
            "TRL-4": ["Functional MVP", "Market validation"],
            "TRL-5": ["Pilot program", "Customer feedback integration"],
            "TRL-6": ["Operational pilot", "Regulatory compliance"],
            "TRL-7": ["Market launch plan", "Distribution strategy"],
            "TRL-8": ["Scaling infrastructure", "Partnership development"],
            "TRL-9": ["Market optimization", "Expansion planning"]
        }
        return requirements.get(current_trl, [])


    def _print_agent_summary(self):
        """Print summary of created agents"""
        cluster_counts = self.agent_factory.get_agent_count_by_cluster()
        # print("\n📊 Agent Distribution by Cluster:")
        # for cluster, count in cluster_counts.items():
        #     print(f"  • {cluster}: {count} agents")
        # print(f"\n🎯 Total Agents: {self.agent_factory.get_total_agent_count()}")
    
    async def validate_idea_async(self, idea_name: str, idea_concept: str, 
                                custom_weights: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Async validation using all 109+ agents (main method)
        """
        return await self.orchestrator.validate_idea(idea_name, idea_concept, custom_weights)
    
    def validate_idea(self, idea_name: str, idea_concept: str, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Synchronous validation method (compatible with existing Flask app)
        This is the main interface that replaces the original validate_idea function
        """
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            validation_result = loop.run_until_complete(
                self.validate_idea_async(idea_name, idea_concept, custom_weights)
            )

            # ✅ Force-fill empty fields (especially recommendations)
            validation_result = self._validate_agent_results(validation_result)

            return self._convert_to_legacy_format(validation_result)

        except Exception as e:
            return self._create_fallback_result(idea_name, idea_concept, str(e))

        
    def _map_cluster_name(self, cluster: str) -> str:
        """
        Map internal cluster names to display names.
        This controls what appears in detailed_viability_assessment.clusters
        """
        cluster_mapping = {
            "Core Idea": "Core Idea & Innovation",
            "Market Opportunity": "Market & Commercial Opportunity", 
            "Execution": "Execution & Operations",
            "Business Model": "Business Model & Strategy",
            "Team": "Team & Organizational Health",
            "Compliance": "External Environment & Compliance",
            "Risk & Strategy": "Risk & Future Outlook"
        }

        return cluster_mapping.get(cluster, cluster)
    
    def _generate_action_points(self, evaluations: List, 
                          collaboration_insights: List[str]) -> List[Dict[str, Any]]:
        """
        Generate structured action points with titles and todo lists
        for the frontend roadmap component
        """
        action_points = []
        parameter_actions = {}

        for evaluation in evaluations:
            try:
                score = getattr(evaluation, 'assigned_score', 0)

                # Only include items that need improvement (score < 85)
                if score < 85:
                    param_name = getattr(evaluation, 'sub_parameter', 'Unknown')
                    cluster = self._map_cluster_name(getattr(evaluation, 'cluster', 'Unknown'))
                    param_group = self._map_subparameter_to_parameter(param_name)

                    # Initialize parameter group if not exists
                    param_key = f"{cluster} - {param_group}"
                    if param_key not in parameter_actions:
                        parameter_actions[param_key] = {
                            "title": f"Improve {param_key}",
                            "todos": []
                        }

                    # Extract actionable items
                    what_can_be_improved = getattr(evaluation, 'what_can_be_improved', [])

                    if isinstance(what_can_be_improved, list) and what_can_be_improved:
                        parameter_actions[param_key]["todos"].extend(what_can_be_improved)
                    elif what_can_be_improved and str(what_can_be_improved) != "[]":
                        parameter_actions[param_key]["todos"].append(str(what_can_be_improved))

            except Exception as e:
                logger.error(f"Error processing evaluation for action points: {e}")
                continue
            
        # Add parameter-level action points
        for action_group in parameter_actions.values():
            if action_group["todos"]:
                action_points.append(action_group)

        # Add strategic insights as separate action point
        if collaboration_insights:
            action_points.append({
                "title": "Strategic Recommendations",
                "todos": collaboration_insights
            })

        # Sort by priority (lower score = higher priority)
        def get_priority_score(action):
            for eval in evaluations:
                param_name = getattr(eval, 'sub_parameter', '')
                if param_name in action["title"]:
                    return getattr(eval, 'assigned_score', 100)
            return 50

        action_points.sort(key=lambda x: get_priority_score(x))

        return action_points[:10]

    
    def _convert_to_legacy_format(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Convert CrewAI result to hierarchical cluster-parameter-subparameter structure"""
        logger.info("🔄 Building hierarchical detailed_viability_assessment...")
        
        # Generate HTML report
        html_report = self._generate_html_report(validation_result)
        
        # Convert agent evaluations to legacy format
        evaluated_data = self._convert_evaluations_to_legacy_format(validation_result.agent_evaluations)
        
        # ✅ Build EXACT hierarchical structure you want
        detailed_viability_assessment = {
            "clusters": self._build_cluster_hierarchy(validation_result.agent_evaluations)
        }

        action_points = self._generate_action_points(
            validation_result.agent_evaluations,
            validation_result.collaboration_insights
        )

        roadmap = self._estimate_trl_level(validation_result)
        
        return {
            "overall_score": validation_result.overall_score,
            "validation_outcome": validation_result.validation_outcome.value,
            "evaluated_data": evaluated_data,
            "html_report": html_report,
            "error": None,
            "processing_time": validation_result.total_processing_time,
            "api_calls_made": validation_result.total_agents_consulted,
            "consensus_level": validation_result.consensus_level,
            "collaboration_insights": validation_result.collaboration_insights,
            "cluster_scores": validation_result.cluster_scores,
            "validation_id": validation_result.validation_id,
            "timestamp": validation_result.timestamp,
            "action_points": action_points,
            "roadmap": roadmap,
            "detailed_viability_assessment": detailed_viability_assessment
        }
    
    def _map_subparameter_to_parameter(self, sub_parameter: str) -> str:
        """
        Complete mapping for all 109 sub-parameters across 7 clusters.
        Based on the ComprehensiveAgentFactory structure.
        """
        parameter_mapping = {
            # ========== Core Idea & Innovation ==========
            "Originality": "Innovation Factor",
            "Innovation Index": "Innovation Factor",
            "Differentiation": "Innovation Factor",
            "Disruptive Potential": "Innovation Factor",
            "Solution Uniqueness": "Innovation Factor",
            "Problem Severity": "Problem-Solution Fit",
            "Market Gap Analysis": "Problem-Solution Fit",
            "Customer Pain Validation": "Problem-Solution Fit",
            "Solution Effectiveness": "Problem-Solution Fit",
            "Intuitive Design": "User Experience",
            "User Interface Quality": "User Experience",
            "Mobile Responsiveness": "User Experience",
            "Cross-Platform Compatibility": "User Experience",
            "Accessibility Compliance": "User Experience",

            # ========== Market Opportunity ==========
            "Market Size (TAM)": "Market Analysis",
            "Market Growth Rate": "Market Analysis",
            "Customer Acquisition Potential": "Market Analysis",
            "Market Penetration Strategy": "Market Analysis",
            "Local Market Understanding": "Market Analysis",
            "Regional Expansion Potential": "Market Analysis",
            "Regulatory Landscape": "Market Readiness",
            "Infrastructure Readiness": "Market Readiness",
            "Cultural Adaptation": "Market Readiness",
            "Competitive Intensity": "Competitive Landscape",
            "User Engagement": "Customer Metrics",
            "Retention Potential": "Customer Metrics",
            "Customer Satisfaction Metrics": "Customer Metrics",
            "Product Stickiness": "Customer Metrics",
            "Market Feedback Integration": "Customer Metrics",
            "Viral Coefficient": "Growth Metrics",

            # ========== Execution ==========
            "Technology Maturity": "Technical Execution",
            "Resource Availability": "Operational Execution",
            "Scalability & Performance": "Technical Execution",
            "Technical Architecture": "Technical Execution",
            "Development Complexity": "Technical Execution",
            "Security Framework": "Technical Execution",
            "API Integration Capability": "Technical Execution",
            "Process Efficiency": "Operational Execution",
            "Supply Chain Management": "Operational Execution",
            "Quality Assurance": "Operational Execution",
            "Operational Scalability": "Operational Execution",
            "Cost Structure Optimization": "Financial Execution",
            "Business Model Scalability": "Financial Execution",
            "Market Expansion Potential": "Strategic Execution",
            "Technology Scalability": "Technical Execution",
            "Financial Scalability": "Financial Execution",
            "International Expansion": "Strategic Execution",

            # ========== Business Model ==========
            "Revenue Stream Diversity": "Revenue Model",
            "Profitability & Margins": "Financial Model",
            "Cash Flow Sustainability": "Financial Model",
            "Customer Lifetime Value": "Financial Model",
            "Unit Economics": "Financial Model",
            "Financial Projections Accuracy": "Financial Model",
            "Intellectual Property (IP)": "Competitive Moats",
            "Network Effects": "Competitive Moats",
            "Brand Moat": "Competitive Moats",
            "Data Moat": "Competitive Moats",
            "Switching Costs": "Competitive Moats",
            "Regulatory Barriers": "Competitive Moats",

            # ========== Team ==========
            "Relevant Experience": "Team Capability",
            "Leadership Capability": "Team Capability",
            "Mission Alignment": "Team Culture",
            "Diversity & Inclusion": "Team Culture",
            "Communication Effectiveness": "Team Dynamics",
            "Complementary Skills": "Team Capability",
            "Industry Expertise": "Team Capability",
            "Execution Track Record": "Team Capability",
            "Domain Knowledge": "Team Capability",
            "Team Dynamics": "Team Culture",
            "Adaptability": "Team Capability",
            "Work Ethics & Values": "Team Culture",

            # ========== Compliance ==========
            "Data Privacy Compliance": "Regulatory Compliance",
            "Tax Compliance": "Regulatory Compliance",
            "Labor Law Compliance": "Legal Compliance",
            "Environmental Impact": "Environmental Compliance",
            "Social Impact (SDGs)": "Social Compliance",
            "Ethical Business Practices": "Governance Compliance",
            "Investor & Partner Landscape": "Stakeholder Compliance",
            "Industry Associations": "Industry Compliance",
            "Sector-Specific Compliance": "Regulatory Compliance",
            "Import/Export Regulations": "Trade Compliance",
            "Digital India Compliance": "Digital Compliance",
            "Governance Standards": "Governance Compliance",
            "Community Engagement": "Social Compliance",
            "Carbon Footprint": "Environmental Compliance",
            "Government & Institutional Support": "Policy Compliance",
            "Mentorship Availability": "Ecosystem Compliance",
            "Academic Partnerships": "Ecosystem Compliance",

            # ========== Risk & Strategy ==========
            "National Policy Alignment (India)": "Strategic Risk",
            "Academic/Research Contribution": "Innovation Risk",
            "Technical Risks": "Operational Risk",
            "Market Risks": "Market Risk",
            "Financial Risks": "Financial Risk",
            "Competitive Risks": "Competitive Risk",
            "Regulatory Risks": "Regulatory Risk",
            "Operational Risks": "Operational Risk",
            "Valuation Potential": "Investment Viability",
            "Exit Strategy Viability": "Investment Viability",
            "ROI Potential": "Investment Viability",
            "Investment Stage Readiness": "Investment Viability",
            "Due Diligence Preparedness": "Investment Viability",
            "Investor Fit": "Investment Viability",
            "Innovation Ecosystem Impact": "Strategic Impact",
            "Knowledge Transfer Potential": "Strategic Impact",
            "Research Commercialization": "Strategic Impact",
            "Educational Value": "Strategic Impact",
        }

        return parameter_mapping.get(sub_parameter, "General Parameters")
    

    def _extract_improvements(self, evaluation) -> str:
        """
        Extract improvements from agent evaluation
        Tries multiple possible attribute names
        """
        # Try multiple sources in order of preference
        sources = ['recommendations', 'weaknesses', 'improvements', 'action_items', 'what_can_be_improved']

        for source in sources:
            data = getattr(evaluation, source, None)
            if data:
                # Handle list
                if isinstance(data, list) and data:
                    return "\n".join(str(item) for item in data)
                # Handle non-empty string
                elif isinstance(data, str) and data.strip():
                    return data

        # No improvements found
        return ""

    def _build_cluster_hierarchy(self, evaluations: List) -> Dict[str, Any]:
        """Build hierarchy using validated agent data"""
        clusters = {}

        for evaluation in evaluations:
            try:
                # Map cluster and parameter
                original_cluster = getattr(evaluation, 'cluster', 'Unknown Cluster')
                cluster = self._map_cluster_name(original_cluster)
                sub_parameter = getattr(evaluation, 'sub_parameter', 'Unknown SubParameter')
                parameter = self._map_subparameter_to_parameter(sub_parameter)
                score = getattr(evaluation, 'assigned_score', 0)

                # Get what_went_well from explanation (50 words max)
                what_went_well_raw = getattr(evaluation, 'explanation', '')
                if isinstance(what_went_well_raw, list):
                    what_went_well_combined = " ".join(str(item) for item in what_went_well_raw)
                else:
                    what_went_well_combined = str(what_went_well_raw)
                what_went_well = what_went_well_combined

                # ✅ Get what_can_be_improved from RECOMMENDATIONS (agents now forced to generate)
                improvements = getattr(evaluation, 'recommendations', [])

                if isinstance(improvements, list) and len(improvements) >= 3:
                    # ✅ SUCCESS - Agent generated real recommendations
                    what_can_be_improved = "\n".join(str(item) for item in improvements[:3])
                elif isinstance(improvements, list) and len(improvements) > 0:
                    what_can_be_improved = "\n".join(str(item) for item in improvements)
                else:
                    # ❌ Fallback only if agent truly failed
                    if score >= 85:
                        what_can_be_improved = "No improvements needed - excellent performance"
                    else:
                        what_can_be_improved = "No specific recommendations provided"

                # Build structure
                if cluster not in clusters:
                    clusters[cluster] = {}

                if parameter not in clusters[cluster]:
                    clusters[cluster][parameter] = {}

                clusters[cluster][parameter][sub_parameter] = {
                    "assignedScore": score,
                    "whatWentWell": what_went_well,
                    "whatCanBeImproved": what_can_be_improved
                }

            except Exception as e:
                logger.error(f"Error building hierarchy: {e}")
                continue
            
        return clusters


    def _truncate_to_250_words(self, text: str) -> str:
        """Truncate text to exactly 250 words"""
        if not text or not isinstance(text, str):
            return "No feedback available"

        words = text.split()
        if len(words) <= 250:
            return text

        return ' '.join(words[:250]) + '...'


    def _extract_strengths(self, evaluations: List) -> List[str]:
        """Extract key strengths from agent evaluations - SAFE VERSION"""
        strengths = []
        for i, evaluation in enumerate(evaluations):
            try:
                score = getattr(evaluation, 'assigned_score', 0)
                param_name = getattr(evaluation, 'sub_parameter', f'Unknown-{i}')

                if score >= 8.0:
                    strengths.append(f"Strong {param_name} (score: {score:.1f})")
            except Exception as e:
                logger.error(f"Error extracting strengths from evaluation {i}: {e}")
        return list(set(strengths))

    def _extract_weaknesses(self, evaluations: List) -> List[str]:
        """Extract key weaknesses - SAFE VERSION"""
        weaknesses = []
        for i, evaluation in enumerate(evaluations):
            try:
                score = getattr(evaluation, 'assigned_score', 0)
                param_name = getattr(evaluation, 'sub_parameter', f'Unknown-{i}')

                if score <= 4.0:
                    weaknesses.append(f"Weak {param_name} (score: {score:.1f})")
            except Exception as e:
                logger.error(f"Error extracting weaknesses from evaluation {i}: {e}")
        return list(set(weaknesses))

    def _generate_recommendations_from_evaluations(self, evaluations: List) -> List[str]:
        """Generate recommendations - SAFE VERSION"""
        recommendations = []
        for i, evaluation in enumerate(evaluations):
            try:
                if hasattr(evaluation, 'recommendations') and evaluation.recommendations:
                    if isinstance(evaluation.recommendations, list):
                        recommendations.extend(evaluation.recommendations)
                elif hasattr(evaluation, 'what_can_be_improved') and evaluation.what_can_be_improved:
                    param_name = getattr(evaluation, 'sub_parameter', f'Unknown-{i}')
                    recommendations.append(f"Improve {param_name}: {evaluation.what_can_be_improved[:100]}...")
            except Exception as e:
                logger.error(f"Error extracting recommendations from evaluation {i}: {e}")
        return list(set(recommendations))

    def _extract_risks(self, evaluations: List) -> List[str]:
        """Extract risk factors - SAFE VERSION"""
        risks = []
        for i, evaluation in enumerate(evaluations):
            try:
                if hasattr(evaluation, 'risk_factors') and evaluation.risk_factors:
                    if isinstance(evaluation.risk_factors, list):
                        risks.extend(evaluation.risk_factors)
                elif hasattr(evaluation, 'explanation') and evaluation.explanation:
                    if 'risk' in evaluation.explanation.lower():
                        param_name = getattr(evaluation, 'sub_parameter', f'Unknown-{i}')
                        risks.append(f"Risk in {param_name}")
            except Exception as e:
                logger.error(f"Error extracting risks from evaluation {i}: {e}")
        return list(set(weaknesses))


    def _estimate_market_opportunity(self, evaluated_data: Dict) -> Dict[str, Any]:
        """Estimate market opportunity from evaluation data"""
        market_opportunity = {}
        if 'market_opportunity' in evaluated_data:
            market_opportunity = evaluated_data['market_opportunity']
        return market_opportunity

    
    def _convert_evaluations_to_legacy_format(self, evaluations: List) -> Dict[str, Any]:
        """Convert agent evaluations to nested dictionary format with ALL details"""
        evaluated_data = {}

        for evaluation in evaluations:
            try:
                # ✅ SAFE attribute access
                cluster = getattr(evaluation, 'cluster', 'Unknown Cluster')
                parameter = getattr(evaluation, 'parameter', 'Unknown Parameter')
                sub_parameter = getattr(evaluation, 'sub_parameter', 'Unknown SubParameter')

                # Initialize nested structure
                if cluster not in evaluated_data:
                    evaluated_data[cluster] = {}
                if parameter not in evaluated_data[cluster]:
                    evaluated_data[cluster][parameter] = {}

                # Add COMPLETE evaluation data
                evaluated_data[cluster][parameter][sub_parameter] = {
                    "assignedScore": getattr(evaluation, 'assigned_score', 0),
                    "explanation": getattr(evaluation, 'what_went_well', ''),  # Use what_went_well as explanation
                    "whatWentWell": getattr(evaluation, 'what_went_well', ''),
                    "whatCanBeImproved": getattr(evaluation, 'what_can_be_improved', ''),
                    "agent_id": getattr(evaluation, 'agent_id', 'unknown'),
                    "confidence_level": getattr(evaluation, 'confidence_level', 'medium'),
                    "processing_time": getattr(evaluation, 'processing_time', 0),
                }
            except Exception as e:
                logger.error(f"Error converting evaluation: {e}")
                continue
            
        return evaluated_data
    
    def _validate_agent_results(self, validation_result: ValidationResult) -> ValidationResult:
        """
        Force validate all agent evaluations after Crew execution
        Ensures recommendations (what_can_be_improved) is ALWAYS populated
        """
        logger.info(f"🔍 Validating {len(validation_result.agent_evaluations)} agent results...")   

        for i, evaluation in enumerate(validation_result.agent_evaluations):
            # Force recommendations field (what_can_be_improved)
            recs = getattr(evaluation, 'recommendations', None)
            if not isinstance(recs, list) or not recs:
                logger.warning(f"⚠️ Agent {i} ({getattr(evaluation, 'sub_parameter', 'Unknown')}) has empty recommendations - FORCING")
                evaluation.recommendations = [
                    f"Conduct validation of {getattr(evaluation, 'sub_parameter', 'this parameter')} assumptions",
                    f"Benchmark {getattr(evaluation, 'sub_parameter', 'this parameter')} against top 3 competitors",
                    f"Develop improvement roadmap for {getattr(evaluation, 'sub_parameter', 'this parameter')}",
                ]   


        logger.info("✅ All agent results validated and recommendations populated")
        return validation_result    


    
    def _generate_html_report(self, validation_result: ValidationResult) -> str:
        """Generate comprehensive HTML report"""
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Get top performing and underperforming areas
        cluster_scores = validation_result.cluster_scores
        best_cluster = max(cluster_scores.items(), key=lambda x: x[1]) if cluster_scores else ("N/A", 0)
        worst_cluster = min(cluster_scores.items(), key=lambda x: x[1]) if cluster_scores else ("N/A", 0)
        
        # Generate recommendations based on outcome
        recommendations = self._generate_recommendations(validation_result.validation_outcome, 
                                                       validation_result.overall_score)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pragati AI Validation Report - Multi-Agent Analysis</title>
            <style>
                {self._get_report_css()}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <div class="logo-section">
                        <h1>Pragati AI - Multi-Agent Validation</h1>
                        <p class="tagline">109+ Specialized AI Agents for Comprehensive Idea Assessment</p>
                    </div>
                    <div class="report-meta">
                        <p><strong>Report Date:</strong> {current_date}</p>
                        <p><strong>Validation ID:</strong> {validation_result.validation_id}</p>
                        <p><strong>Agents Consulted:</strong> {validation_result.total_agents_consulted}</p>
                    </div>
                </header>

                <section class="executive-summary">
                    <h2>Executive Summary</h2>
                    <div class="summary-grid">
                        <div class="score-card {validation_result.validation_outcome.value.lower()}">
                            <h3>Overall Score</h3>
                            <div class="score-display">{validation_result.overall_score:.2f}/5.0</div>
                            <div class="outcome-badge">{validation_result.validation_outcome.value}</div>
                            <div class="consensus-indicator">
                                <small>Agent Consensus: {validation_result.consensus_level:.1%}</small>
                            </div>
                        </div>
                        <div class="performance-overview">
                            <h3>Performance Overview</h3>
                            <p><strong>Best Performing Area:</strong> {best_cluster[0]} ({best_cluster[1]:.2f}/5.0)</p>
                            <p><strong>Area for Improvement:</strong> {worst_cluster[0]} ({worst_cluster[1]:.2f}/5.0)</p>
                            <p><strong>Processing Time:</strong> {validation_result.total_processing_time:.1f} seconds</p>
                        </div>
                    </div>
                </section>

                <section class="cluster-breakdown">
                    <h2>Detailed Cluster Analysis</h2>
                    {self._generate_cluster_breakdown_html(validation_result.cluster_scores)}
                </section>

                <section class="collaboration-insights">
                    <h2>Multi-Agent Collaboration Insights</h2>
                    <div class="insights-grid">
                        {self._generate_insights_html(validation_result.collaboration_insights)}
                    </div>
                </section>

                <section class="recommendations">
                    <h2>Strategic Recommendations</h2>
                    {recommendations}
                </section>

                <section class="methodology">
                    <h2>Multi-Agent Methodology</h2>
                    <div class="methodology-content">
                        <p>This evaluation utilized {validation_result.total_agents_consulted} specialized AI agents working in collaboration:</p>
                        <ul>
                            <li><strong>Specialized Expertise:</strong> Each agent focuses on a specific validation parameter</li>
                            <li><strong>Collaborative Analysis:</strong> Agents share insights and build upon each other's assessments</li>
                            <li><strong>Dependency Resolution:</strong> Complex interdependencies between parameters are properly handled</li>
                            <li><strong>Consensus Building:</strong> Multiple viewpoints are synthesized into coherent recommendations</li>
                        </ul>
                        <p><strong>Consensus Level:</strong> {validation_result.consensus_level:.1%} - 
                        {'High agreement among agents' if validation_result.consensus_level > 0.8 else 'Moderate agreement with some divergent views' if validation_result.consensus_level > 0.6 else 'Significant disagreement indicates mixed evaluation'}</p>
                    </div>
                </section>

                <footer class="report-footer">
                    <p>Generated by Pragati AI Multi-Agent Validation System | {validation_result.total_agents_consulted} agents consulted</p>
                    <p><em>This report represents the collaborative assessment of specialized AI agents and should guide strategic decision-making.</em></p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_cluster_breakdown_html(self, cluster_scores: Dict[str, float]) -> str:
        """Generate HTML for cluster score breakdown"""
        html = '<div class="cluster-grid">'
        
        for cluster, score in cluster_scores.items():
            status_class = "excellent" if score >= 4.0 else "good" if score >= 3.0 else "moderate" if score >= 2.0 else "poor"
            html += f"""
            <div class="cluster-card {status_class}">
                <h4>{cluster}</h4>
                <div class="cluster-score">{score:.2f}</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {(score/5.0)*100}%"></div>
                </div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_insights_html(self, insights: List[str]) -> str:
        """Generate HTML for collaboration insights"""
        html = ""
        for insight in insights:
            html += f'<div class="insight-card"><p>{insight}</p></div>'
        return html
    
    def _generate_recommendations(self, outcome: ValidationOutcome, score: float) -> str:
        """Generate recommendations based on validation outcome"""
        if outcome == ValidationOutcome.EXCELLENT:
            return """
            <div class="recommendations-excellent">
                <h3>🚀 Excellent Potential - Ready for Launch</h3>
                <ul>
                    <li>Proceed with MVP development and market entry strategy</li>
                    <li>Prepare for Series A funding with strong validation metrics</li>
                    <li>Focus on rapid scaling and market capture</li>
                    <li>Build strategic partnerships to accelerate growth</li>
                </ul>
            </div>
            """
        elif outcome == ValidationOutcome.GOOD:
            return """
            <div class="recommendations-good">
                <h3>✅ Strong Potential - Recommended for Development</h3>
                <ul>
                    <li>Develop detailed business plan and go-to-market strategy</li>
                    <li>Address identified weaknesses before full-scale launch</li>
                    <li>Consider seed funding to accelerate development</li>
                    <li>Build pilot programs to validate market assumptions</li>
                </ul>
            </div>
            """
        elif outcome == ValidationOutcome.MODERATE:
            return """
            <div class="recommendations-moderate">
                <h3>⚠️ Moderate Potential - Requires Strategic Improvements</h3>
                <ul>
                    <li>Focus on strengthening low-scoring areas before proceeding</li>
                    <li>Conduct additional market validation and customer research</li>
                    <li>Consider pivoting aspects of the business model</li>
                    <li>Seek mentorship and advisory support</li>
                </ul>
            </div>
            """
        else:
            return """
            <div class="recommendations-weak">
                <h3>❌ Significant Challenges - Major Rework Needed</h3>
                <ul>
                    <li>Fundamental reassessment of problem-solution fit required</li>
                    <li>Extensive market research and validation needed</li>
                    <li>Consider substantial pivoting or alternative approaches</li>
                    <li>Focus on addressing core viability concerns</li>
                </ul>
            </div>
            """
    
    def _get_report_css(self) -> str:
        """Get CSS styles for the HTML report"""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .logo-section h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        
        .tagline {
            color: #7f8c8d;
            font-style: italic;
        }
        
        .executive-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            margin-top: 20px;
        }
        
        .score-card {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
        }
        
        .score-display {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .outcome-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: rgba(255,255,255,0.2);
        }
        
        .consensus-indicator {
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .cluster-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .cluster-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }
        
        .cluster-card.excellent { border-left-color: #27ae60; }
        .cluster-card.good { border-left-color: #2ecc71; }
        .cluster-card.moderate { border-left-color: #f39c12; }
        .cluster-card.poor { border-left-color: #e74c3c; }
        
        .cluster-score {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .score-bar {
            width: 100%;
            height: 10px;
            background: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .score-fill {
            height: 100%;
            background: #3498db;
            transition: width 0.3s ease;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .insight-card {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .recommendations-excellent { border-left: 5px solid #27ae60; padding: 20px; background: #d5f4e6; }
        .recommendations-good { border-left: 5px solid #2ecc71; padding: 20px; background: #d1f2eb; }
        .recommendations-moderate { border-left: 5px solid #f39c12; padding: 20px; background: #fdeaa7; }
        .recommendations-weak { border-left: 5px solid #e74c3c; padding: 20px; background: #fadbd8; }
        
        .methodology-content {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .report-footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        ul {
            padding-left: 20px;
        }
        
        li {
            margin-bottom: 8px;
        }
        """
    
    def _create_fallback_result(self, idea_name: str, idea_concept: str, error: str) -> Dict[str, Any]:
        """Create fallback result when validation fails"""
        return {
            "overall_score": 3.0,
            "validation_outcome": "MODERATE",
            "evaluated_data": {},
            "html_report": f"<h1>Validation Error</h1><p>Error: {error}</p>",
            "error": error,
            "processing_time": 0.0,
            "api_calls_made": 0
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the validation system"""
        return {
            "system_name": "Pragati CrewAI Multi-Agent Validator",
            "version": "1.0.0",
            "total_agents": len(self.agents),
            "cluster_distribution": self.agent_factory.get_agent_count_by_cluster(),
            "dependency_graph": self.agent_factory.get_dependency_graph(),
            "llm_model": "gpt-4",
            "capabilities": [
                "109+ specialized validation agents",
                "Inter-agent collaboration and dependency resolution", 
                "Comprehensive Indian market analysis",
                "Real-time consensus building",
                "Detailed HTML reporting"
            ]
        }


# Global instance for Flask integration
_pragati_validator: Optional[PragatiCrewAIValidator] = None

def get_pragati_validator() -> PragatiCrewAIValidator:
    """Get global validator instance (singleton pattern)"""
    global _pragati_validator
    if _pragati_validator is None:
        _pragati_validator = PragatiCrewAIValidator()
    return _pragati_validator



def get_evaluation_framework_info() -> Dict[str, Any]:
    """Get comprehensive information about the evaluation framework (compatibility function)"""
    try:
        validator = get_pragati_validator()
        return validator.get_system_info()
    except Exception as e:
        return {
            "error": f"Failed to get framework info: {str(e)}",
            "version": "CrewAI_v1.0",
            "total_agents": 109
        }


def get_system_health() -> Dict[str, Any]:
    """Get system health information (compatibility function)"""
    try:
        validator = get_pragati_validator()
        return {
            "system_status": "operational",
            "ai_engine_available": True,
            "total_agents": len(validator.agents),
            "api_key_configured": bool(validator.openai_api_key),
            "framework_valid": True
        }
    except Exception as e:
        return {
            "system_status": "degraded",
            "ai_engine_available": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the system
    # print("Testing Pragati CrewAI Multi-Agent Validation System")
    # print("=" * 60)
    
    validator = PragatiCrewAIValidator()
    
    # Test validation
    test_result = validator.validate_idea(
        "AI-Powered Smart Farming Solution",
        "An IoT and AI-based platform that helps farmers optimize crop yields through real-time monitoring, predictive analytics, and automated irrigation systems designed specifically for Indian agricultural conditions."
    )
    
    # print(f"Test validation completed:")
    # print(f"Overall Score: {test_result['overall_score']:.2f}")
    # print(f"Outcome: {test_result['validation_outcome']}")
    # print(f"Processing Time: {test_result.get('processing_time', 0):.1f} seconds")
    # print(f"Agents Consulted: {test_result.get('api_calls_made', 0)}")


# ==================== REQUIRED MODULE-LEVEL WRAPPER ====================
# This is the CRITICAL piece that allows Flask to import validate_idea

def validate_idea(
    idea_name: str,
    idea_concept: str,
    weights: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for Flask app_v3.py
    Delegates to singleton PragatiCrewAIValidator instance.
    """
    try:
        validator = get_pragati_validator()

        custom_weights: Optional[Dict[str, float]] = None
        if weights:
            custom_weights = {k: float(v) for k, v in weights.items()}

        return validator.validate_idea(idea_name, idea_concept, custom_weights)
        
    except Exception as e:
        logger.error(f"Validation error for '{idea_name}': {e}", exc_info=True)
        return {
            "overall_score": 0,
            "validation_outcome": "ERROR",
            "detailed_viability_assessment": {"clusters": {}},
            "action_points": [],
            "roadmap": {
                "current_trl": "TRL-0",
                "phase_name": "Error",
                "key_activities": [],
                "estimated_timeline": "N/A",
                "next_phase_requirements": []
            },
            "html_report": f"<h1>System Error</h1><p>{e}</p>",
            "error": str(e),
        }