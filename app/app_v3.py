"""
Pragati Backend v3.0 - Integrated with CrewAI Multi-Agent Validation System
Flask application using 109+ specialized AI agents for comprehensive idea validation.
"""
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from bson import ObjectId


# ========================================
# Setup Python Path - CRITICAL!
# ========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# print(f"📁 Python path setup:")
# print(f"   Current: {current_dir}")
# print(f"   Parent: {parent_dir}")

# ========================================
# Standard Library Imports
# ========================================
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, Response
from flask_cors import CORS
import queue
import threading
from dotenv import load_dotenv
import logging

# ========================================
# Local Module Imports - With Error Handling
# ========================================
# print("\n📦 Loading local modules...")

try:
    from pdf_report_system.report_writer import AIReportWriter
    # print("   ✅ AIReportWriter from pdf_report_system")
except ImportError as e:
    print(f"   ❌ Failed to import AIReportWriter: {e}")
    raise

try:
    from crew_ai_integration import validate_idea, get_evaluation_framework_info, get_system_health
    # print("   ✅ crew_ai_integration")
except ImportError as e:
    print(f"   ❌ Failed to import crew_ai_integration: {e}")
    raise

try:
    from pdf_generator import ValidationReportGenerator
    # print("   ✅ pdf_generator")
except ImportError as e:
    print(f"   ❌ Failed to import pdf_generator: {e}")
    raise

try:
    from pitch_deck_processor import PitchDeckProcessor
    # print("   ✅ pitch_deck_processor")
except ImportError as e:
    print(f"   ❌ Failed to import pitch_deck_processor: {e}")
    raise

try:
    from database_manager import get_database_manager
    # print("   ✅ database_manager")
except ImportError as e:
    print(f"   ❌ Failed to import database_manager: {e}")
    raise

try:
    from report_endpoints import register_report_endpoints
    # print("   ✅ report_endpoints")
except ImportError as e:
    print(f"   ❌ Failed to import report_endpoints: {e}")
    raise

try:
    from psychometric_endpoints import register_psychometric_endpoints
    # print("   ✅ psychometric_endpoints")
except ImportError as e:
    print(f"   ❌ Failed to import psychometric_endpoints: {e}")
    raise

try:
    from user_profile_manager import get_user_profile_manager
    # print("   ✅ user_profile_manager")
except ImportError as e:
    print(f"   ❌ Failed to import user_profile_manager: {e}")
    raise

# print("✅ All modules loaded successfully!\n")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

# Ensure the app's logger is also set to INFO
app.logger.setLevel(logging.INFO)


# Initialize PDF generator, pitch deck processor, and database manager
pdf_generator = ValidationReportGenerator()
pitch_deck_processor = PitchDeckProcessor()

# Initialize database manager
try:
    db_manager = get_database_manager()
    logger.info("✅ Database manager initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize database manager: {e}")
    db_manager = None

# Register report management endpoints
register_report_endpoints(app)

# Register psychometric assessment endpoints
register_psychometric_endpoints(app)
logger.info("✅ Psychometric endpoints registered")

# Global queue for real-time agent messages
agent_message_queue = queue.Queue()
active_connections = set()

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_info = get_system_health()
        return jsonify({
            "status": "healthy",
            "message": "Pragati Backend v3.0 with CrewAI Multi-Agent System",
            "system_info": health_info
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "error": str(e)
        }), 500


@app.route('/logos/<path:filename>')
def serve_logo(filename):
    """Serve logo files from the logos directory"""
    logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logos')
    return send_from_directory(logos_dir, filename)


@app.route('/api/validate-idea', methods=['POST'])
def validate_idea_endpoint():
    """
    Main idea validation endpoint using 109+ CrewAI agents
    
    Request body:
    {
        "user_id": "string",
        "title": "string",
        "idea_name": "string",
        "idea_concept": "string", 
        "custom_weights": {optional dict of cluster weights}
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'title', 'idea_name', 'idea_concept']
        missing_fields = [field for field in required_fields if not data or field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        user_id = data['user_id'].strip()
        title = data['title'].strip()
        idea_name = data['idea_name'].strip()
        idea_concept = data['idea_concept'].strip()
        custom_weights = data.get('custom_weights')
        
        if not all([user_id, title, idea_name, idea_concept]):
            return jsonify({
                "error": "All required fields must be non-empty"
            }), 400
        
        logger.info(f"Starting validation for idea: {idea_name} (User: {user_id})")
        
        # Broadcast validation start
        broadcast_agent_message("System", f"🚀 Starting validation for '{idea_name}'", "system")
        
        # Run validation using CrewAI multi-agent system
        try:
            result = validate_idea(idea_name, idea_concept, custom_weights)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return jsonify({
                "error": f"Validation failed: {str(e)}"
            }), 500
        
        # Get user profile for personalized insights (if available)
        profile_manager = get_user_profile_manager()
        user_context = profile_manager.get_personalized_validation_context(user_id)
        
        if user_context.get('has_profile'):
            logger.info(f"📊 Using personalized validation context for user: {user_id}")
            result['personalized_insights'] = {
                "user_fit_score": user_context.get('fit_score'),
                "entrepreneurial_fit": user_context.get('entrepreneurial_fit'),
                "strengths_to_leverage": user_context.get('strengths', [])[:3],
                "areas_to_focus": user_context.get('weak_areas', [])[:3]
            }
        
        # Save to MongoDB
        report_id = None
        if db_manager:
            try:
                # Convert ValidationResult object to dict before saving
                result_dict = result.__dict__ if hasattr(result, '__dict__') else result
        
                # Log the keys for verification
                logger.info(f"Detailed assessment keys: {list(result_dict.get('detailed_viability_assessment', {}).keys())}")
        
                report_id = db_manager.save_validation_report(
                    user_id=user_id,
                    title=title,
                    validation_result=result_dict, # ✅ Now passes dict
                    idea_name=idea_name,
                    idea_concept=idea_concept,
                    source_type="manual",
                    action_points=result_dict.get("action_points")
                )
                logger.info(f"✅ Saved report to MongoDB: {report_id}")
                broadcast_agent_message("System", f"💾 Report saved to database: {report_id}", "success")
                
                # Add to user's validation history
                if user_context.get('has_profile'):
                    profile_manager.add_validation_to_history(
                        user_id=user_id,
                        idea_name=idea_name,
                        validation_result=result,
                        report_id=report_id
                    )
            except Exception as e:
                logger.error(f"❌ Failed to save to MongoDB: {e}")
                # Continue without failing the validation

            if db_manager:
                try:
                    writer = AIReportWriter()
                    ai_report = writer.write_comprehensive_report(
                        result.get('evaluated_data', {}),
                        {'title': idea_name, 'overall_score': result.get('overall_score', 0)}
                    )
                    db_manager.save_ai_report(
                        report_id=report_id,
                        ai_report=ai_report,
                        user_id=user_id,
                        idea_name=idea_name,
                        idea_concept=idea_concept
                    )
                    logger.info("✅ AI report with risk, benchmarking, recommendations saved")
                except Exception as e:
                    logger.error(f"❌ AI report error: {e}")

        # Broadcast validation completion
        overall_score = result.get('overall_score', 3.0)
        broadcast_agent_message("System", f"✅ Validation completed! Overall score: {overall_score:.2f}/5.0", "success")

        # Add report ID to result
        if report_id:
            result['report_id'] = report_id
            result['saved_to_database'] = True
        else:
            result['saved_to_database'] = False

        return jsonify(result)
        
        
    except Exception as e:
        logger.error(f"Validation endpoint error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500
    
@app.route("/api/validate-pitch-decks-batch", methods=["POST"])
def validate_pitch_decks_batch():
    """
    Batch validate multiple pitch decks from ideas collection.
    Each validation result is saved as a separate document in the 'results' collection.
    NOW WITH ALL FIELDS MATCHING SINGLE ENDPOINT (detailed_analysis, roadmap, raw_validation_result, ai_report, evaluated_data)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        idea_ids = data.get("ideaIds", [])
        custom_weights = data.get("customWeights", None)

        if not idea_ids or not isinstance(idea_ids, list) or len(idea_ids) == 0:
            return jsonify({"error": "ideaIds must be a non-empty array"}), 400

        # ✅ CONVERT STRING IDS TO ObjectId
        converted_ids = []
        for id_str in idea_ids:
            try:
                converted_ids.append(ObjectId(id_str))
            except Exception:
                return jsonify({"error": f"Invalid idea ID format: {id_str}"}), 400
        
        log_message = f"🔄 Batch validation started for {len(converted_ids)} ideas"
        app.logger.info(log_message)

        batch_results = []
        successful_count = 0
        failed_count = 0

        for idx, idea_id in enumerate(converted_ids, 1):
            result_item = {
                "ideaId": str(idea_id),  # ✅ FIX 1: Convert ObjectId to string immediately
                "index": idx,
                "total": len(converted_ids),  # ✅ FIXED: Use converted_ids
                "status": "pending",
                "error": None,
                "innovatorId": None,
                "title": None,
                "originalFilename": None,
                "resultDocId": None,
                "savedToDatabase": False
            }

            temp_path = None

            try:
                app.logger.info(f"⏳ Idea {idx}/{len(converted_ids)}: Fetching {idea_id}")

                if not db_manager:
                    raise Exception("Database manager not available")

                ideas_collection = db_manager.db['pragati-innovation-suite_ideas']
                print(idea_id)
                idea_doc = ideas_collection.find_one({"_id": idea_id})

                if not idea_doc:
                    raise Exception(f"Idea not found in database: {idea_id}")

                # Extract idea info
                innovator_id = str(idea_doc.get("innovatorId", "unknown"))
                title = idea_doc.get("title", "Untitled")
                concept = idea_doc.get("concept", "")
                background = idea_doc.get("background", "")
                ppt_file_key = idea_doc.get("pptFileKey", "")
                ppt_file_name = idea_doc.get("pptFileName", "unknown.pptx")
                ppt_file_url = idea_doc.get("pptFileUrl", "")
                ppt_file_size = idea_doc.get("pptFileSize", 0)

                # ✅ FIX 2: Handle missing PPT file gracefully - use concept/background instead
                if not ppt_file_url or not ppt_file_key:
                    app.logger.warning(f"⚠️ No PPT file for idea {idea_id}, using concept/background directly")
                    
                    # Use concept or background for validation
                    extracted_idea_name = title
                    extracted_idea_concept = concept or background or "No detailed concept provided"
                    
                    # Skip PPT processing and go directly to validation
                    app.logger.info(f"🤖 Running AI validation for: {extracted_idea_name}")
                    
                    validation_result = validate_idea(
                        extracted_idea_name,
                        extracted_idea_concept,
                        custom_weights
                    )
                    
                    if not validation_result or validation_result.get("error"):
                        raise Exception(f"Validation failed: {validation_result.get('error', 'Unknown error')}")
                    
                    app.logger.info(f"✅ Validation complete for: {extracted_idea_name}")
                    
                    # Continue to personalized insights section (skip S3/PPT processing)
                    
                else:
                    # Original PPT processing flow
                    result_item["innovatorId"] = innovator_id
                    result_item["title"] = title
                    result_item["originalFilename"] = ppt_file_name

                    app.logger.info(f"✅ Idea fetched: {title} by {innovator_id}")

                    # ========================
                    # DOWNLOAD FROM S3
                    # ========================
                    app.logger.info(f"📥 Downloading from S3: {ppt_file_key}")

                    file_ext = os.path.splitext(ppt_file_name)[1].lower()
                    if file_ext not in [".pdf", ".ppt", ".pptx"]:
                        raise Exception(f"Invalid file extension: {file_ext}")

                    import tempfile
                    import boto3

                    # ✅ Create temp file without closing it prematurely
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=file_ext,
                        prefix="batch_pitch_",
                        mode='w+b'
                    )

                    temp_file.close()  # ✅ Close AFTER creation, before S3 download
                    temp_path = temp_file.name

                    try:
                        s3_client = boto3.client(
                            's3',
                            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                            region_name=os.getenv("AWS_REGION", "us-east-1")
                        )

                        bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "pragati-docs")

                        # ✅ Download to the path directly (file is closed, so no lock)
                        s3_client.download_file(bucket_name, ppt_file_key, temp_path)
                        app.logger.info(f"✅ File downloaded to: {temp_path}")

                    except Exception as s3_err:
                        raise Exception(f"S3 download failed: {str(s3_err)}")

                    # ========================
                    # PROCESS PITCH DECK
                    # ========================
                    app.logger.info(f"🔍 Extracting content from: {ppt_file_name}")

                    try:
                        extracted_info = pitch_deck_processor.process_pitch_deck(temp_path)
                        extracted_idea_name = extracted_info.get("idea_name", title)
                        extracted_idea_concept = extracted_info.get("idea_concept", concept)
                        app.logger.info(f"✅ Extracted: {extracted_idea_name}")

                    except Exception as process_err:
                        app.logger.warning(f"⚠️ Pitch deck extraction warning: {str(process_err)}")
                        extracted_idea_name = title
                        extracted_idea_concept = concept

                    # ========================
                    # RUN AI VALIDATION
                    # ========================
                    app.logger.info(f"🤖 Running AI validation for: {extracted_idea_name}")

                    validation_result = validate_idea(
                        extracted_idea_name,
                        extracted_idea_concept,
                        custom_weights
                    )

                    if not validation_result or validation_result.get("error"):
                        raise Exception(f"Validation failed: {validation_result.get('error', 'Unknown error')}")

                    app.logger.info(f"✅ Validation complete for: {extracted_idea_name}")

                # ========================
                # PERSONALIZED INSIGHTS (✅ ADDED)
                # ========================
                try:
                    profile_manager = get_user_profile_manager()
                    user_context = profile_manager.get_personalized_validation_context(innovator_id)

                    if user_context.get('has_profile'):
                        app.logger.info(f"📊 Adding personalized insights for user: {innovator_id}")
                        validation_result['personalized_insights'] = {
                            "user_fit_score": user_context.get('fit_score'),
                            "entrepreneurial_fit": user_context.get('entrepreneurial_fit'),
                            "strengths_to_leverage": user_context.get('strengths', [])[:3],
                            "areas_to_focus": user_context.get('weak_areas', [])[:3]
                        }
                except Exception as profile_err:
                    app.logger.warning(f"⚠️ Could not fetch personalized insights: {profile_err}")

                # ========================
                # GENERATE DETAILED ANALYSIS (✅ ADDED)
                # ========================
                try:
                    if db_manager:
                        detailed_analysis = db_manager.generate_detailed_report_data(
                            validation_result,
                            extracted_idea_name,
                            extracted_idea_concept
                        )
                        validation_result['detailed_analysis'] = detailed_analysis
                        app.logger.info(f"✅ Detailed analysis generated for: {extracted_idea_name}")
                except Exception as detail_err:
                    app.logger.error(f"❌ Failed to generate detailed analysis: {detail_err}")
                    validation_result['detailed_analysis'] = {}

                # ========================
                # NORMALIZE & ENRICH VALIDATION RESULT (✅ CRITICAL FIX)
                # ========================

                # Extract all required fields from validation result
                overall_score = validation_result.get("overall_score", 0)
                validation_outcome = validation_result.get("validation_outcome", "UNKNOWN")

                # Ensure all fields that single endpoint generates
                normalized_result = {
                    "overall_score": overall_score,
                    "validation_outcome": validation_outcome,
                    "evaluated_data": validation_result.get("evaluated_data", {}),
                    "action_points": validation_result.get("action_points", []),
                    "total_agents_consulted": validation_result.get("total_agents_consulted", 0),
                    "api_calls_made": validation_result.get("api_calls_made", 0),
                    "consensus_level": validation_result.get("consensus_level", 0),
                    "processing_time": validation_result.get("processing_time", 0),

                    # ✅ ADD MISSING NESTED FIELDS
                    "detailed_viability_assessment": validation_result.get("detailed_viability_assessment", {}),
                    "detailed_analysis": validation_result.get("detailed_analysis", {}),
                    "roadmap": validation_result.get("roadmap", {}),
                    "raw_validation_result": validation_result.get("raw_validation_result", {}),
                    "ai_report": validation_result.get("ai_report", ""),
                    "personalized_insights": validation_result.get("personalized_insights"),  # ✅ Added explicitly

                    # Keep all other fields from original result
                    **{k: v for k, v in validation_result.items() if k not in [
                        "overall_score", "validation_outcome", "evaluated_data",
                        "action_points", "total_agents_consulted", "api_calls_made",
                        "consensus_level", "processing_time", "detailed_viability_assessment",
                        "detailed_analysis", "roadmap", "raw_validation_result", "ai_report",
                        "personalized_insights"
                    ]}
                }

                # Update to use normalized result
                validation_result = normalized_result

                # ✅ Merge ALL validation_result keys (snake_case) into result_item
                # This ensures frontend gets exactly what validate-idea returns
                result_item.update(validation_result)

                # ========================
                # SAVE RESULT TO DATABASE
                # ========================
                app.logger.info(f"💾 Saving result document: {extracted_idea_name}")

                if db_manager:
                    try:
                        result_doc = {
                            "ideaId": str(idea_id),  # ✅ FIXED: Convert ObjectId to string for JSON serialization
                            "innovatorId": innovator_id,
                            "title": title,
                            "concept": concept,
                            "background": background,
                            "originalFilename": ppt_file_name,
                            "pptFileKey": ppt_file_key,
                            "pptFileSize": ppt_file_size,
                            "source_type": "pitch_deck_batch",
                            "extractedIdeaName": extracted_idea_name,
                            "extractedIdeaConcept": extracted_idea_concept,

                            # ✅ FULL NORMALIZED VALIDATION RESULT
                            "validationResult": validation_result,

                            # ✅ All top-level fields (matching single endpoint)
                            "overallScore": overall_score,
                            "validationOutcome": validation_outcome,
                            "totalAgentsConsulted": validation_result.get("total_agents_consulted", 0),
                            "apiCallsMade": validation_result.get("api_calls_made", 0),
                            "consensusLevel": validation_result.get("consensus_level", 0),
                            "processingTime": validation_result.get("processing_time", 0),

                            # ✅ ALL MISSING NESTED FIELDS (NOW INCLUDED)
                            "actionPoints": validation_result.get("action_points", []),
                            "detailedViabilityAssessment": validation_result.get("detailed_viability_assessment", {}),
                            "detailedAnalysis": validation_result.get("detailed_analysis", {}),
                            "roadmap": validation_result.get("roadmap", {}),
                            "rawValidationResult": validation_result.get("raw_validation_result", {}),
                            "aiReport": validation_result.get("ai_report", ""),
                            "evaluatedData": validation_result.get("evaluated_data", {}),

                            # Metadata
                            "validationTimestamp": datetime.utcnow().isoformat(),
                            "batchProcessing": True,
                            "batchIndex": idx,
                            "totalInBatch": len(converted_ids),  # ✅ FIXED: Use converted_ids
                            "createdAt": datetime.utcnow(),
                            "updatedAt": datetime.utcnow()
                        }

                        results_collection = db_manager.db['results']
                        insert_result = results_collection.insert_one(result_doc)
                        result_doc_id = str(insert_result.inserted_id)
                        result_item["resultDocId"] = result_doc_id
                        result_item["report_id"] = result_doc_id # ✅ Added for consistency with validate-idea
                        result_item["savedToDatabase"] = True
                        app.logger.info(f"✅ Result document saved with ID: {result_doc_id}")

                        # Update idea document
                        ideas_collection.update_one(
                            {"_id": idea_id},
                            {"$set": {
                                "overallScore": overall_score,
                                "status": validation_outcome,
                                "resultDocId": result_doc_id,
                                "validationTimestamp": datetime.utcnow().isoformat(),
                            }}
                        )

                        app.logger.info(f"✅ Idea updated: overallScore={overall_score}, status={validation_outcome}")

                    except Exception as db_err:
                        app.logger.error(f"❌ Database save error: {str(db_err)}")
                        raise Exception(f"Failed to save result: {str(db_err)}")

                # ========================
                # CLEANUP TEMP FILE
                # ========================
                if temp_path:
                    try:
                        os.unlink(temp_path)
                        app.logger.info(f"✅ Temp file cleaned: {temp_path}")
                    except Exception as cleanup_err:
                        app.logger.warning(f"⚠️ Cleanup warning: {str(cleanup_err)}")

                result_item["status"] = "success"
                successful_count += 1
                app.logger.info(f"✅ Idea {idx}/{len(converted_ids)} completed successfully")  # ✅ FIXED: Use converted_ids

            except Exception as item_err:
                error_msg = str(item_err)
                result_item["status"] = "failed"
                result_item["error"] = error_msg
                failed_count += 1
                app.logger.error(f"❌ Error processing idea {idea_id}: {error_msg}")

                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            batch_results.append(result_item)

        summary = {
            "total_ideas": len(converted_ids),  # ✅ FIXED: Use converted_ids
            "successful": successful_count,
            "failed": failed_count,
            "timestamp": datetime.utcnow().isoformat(),
            "results": batch_results
        }

        final_msg = f"✅ Batch validation complete: {successful_count}/{len(converted_ids)} successful"  # ✅ FIXED: Use converted_ids
        app.logger.info(final_msg)

        return jsonify(summary), 200

    except Exception as e:
        error_msg = f"Batch validation fatal error: {str(e)}"
        app.logger.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/framework-info', methods=['GET'])
def framework_info():
    """Get information about the evaluation framework"""
    try:
        framework_info = get_evaluation_framework_info()
        return jsonify({
            "success": True,
            "data": framework_info
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to get framework info",
            "details": str(e)
        }), 500


@app.route('/api/system-info', methods=['GET'])
def system_info():
    """Get comprehensive system information"""
    try:
        from crew_ai_integration import get_pragati_validator
        
        validator = get_pragati_validator()
        system_info = validator.get_system_info()
        
        return jsonify({
            "success": True,
            "data": system_info
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Failed to get system info",
            "details": str(e)
        }), 500


@app.route('/api/agents', methods=['GET'])
def get_agents_info():
    """Get information about all validation agents"""
    try:
        from crew_ai_integration import get_pragati_validator
        
        validator = get_pragati_validator()
        
        agents_info = {}
        for agent_id, agent in validator.agents.items():
            agents_info[agent_id] = {
                "cluster": agent.cluster,
                "parameter": agent.parameter,
                "sub_parameter": agent.sub_parameter,
                "weight": agent.weight,
                "dependencies": agent.dependencies
            }
        
        return jsonify({
            "success": True,
            "total_agents": len(agents_info),
            "agents": agents_info,
            "cluster_distribution": validator.agent_factory.get_agent_count_by_cluster()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Failed to get agents info",
            "details": str(e)
        }), 500


@app.route('/api/validate-pitch-deck', methods=['POST'])
def validate_pitch_deck():
    """
    Validate idea from database using ideaId
    
    Request: JSON body with:
    - 'ideaId' string (MongoDB ObjectId)
    Optional: 'customWeights' dict
    """
    print("\n" + "="*80)
    print("🚀 VALIDATE-PITCH-DECK ENDPOINT CALLED")
    print("="*80)
    
    try:
        # Step 1: Get JSON data
        print("\n[STEP 1] 📥 Getting request JSON data...")
        data = request.get_json()
        
        if not data:
            print("❌ No JSON data received")
            return jsonify({"error": "Request body must be JSON"}), 400
        
        print(f"✅ Request data received")
        print(f"   Keys in request: {list(data.keys())}")
        
        # Step 2: Extract ideaId
        print("\n[STEP 2] 🔑 Extracting ideaId...")
        idea_id_str = data.get("ideaId")
        custom_weights = data.get("customWeights", None)
        
        if not idea_id_str:
            print("❌ ideaId missing")
            return jsonify({"error": "ideaId is required"}), 400
        
        print(f"   ideaId: {idea_id_str}")
        print(f"   customWeights: {custom_weights is not None}")
        
        # Step 3: Convert ideaId to ObjectId
        print("\n[STEP 3] 🔄 Converting ideaId to ObjectId...")
        try:
            idea_id = ObjectId(idea_id_str)
            print(f"✅ Converted to ObjectId: {idea_id}")
        except Exception as conv_err:
            print(f"❌ Invalid ideaId format: {conv_err}")
            return jsonify({"error": f"Invalid ideaId format: {idea_id_str}"}), 400
        
        # Step 4: Fetch idea from database
        print("\n[STEP 4] 📊 Fetching idea from database...")
        
        if not db_manager:
            print("❌ Database manager not available")
            return jsonify({"error": "Database manager not available"}), 500
        
        ideas_collection = db_manager.db['pragati-innovation-suite_ideas']
        
        try:
            idea_doc = ideas_collection.find_one({"_id": idea_id})
            
            if not idea_doc:
                print(f"❌ Idea not found: {idea_id}")
                return jsonify({"error": f"Idea not found: {idea_id_str}"}), 404
            
            print("✅ Idea document found")
            
        except Exception as db_err:
            print(f"❌ Database query failed: {db_err}")
            return jsonify({"error": f"Database query failed: {str(db_err)}"}), 500
        
        # Step 5: Extract idea details
        print("\n[STEP 5] 📝 Extracting idea details...")
        innovator_id = str(idea_doc.get("innovatorId", "unknown"))
        title = idea_doc.get("title", "Untitled")
        concept = idea_doc.get("concept", "")
        background = idea_doc.get("background", "")
        ppt_file_key = idea_doc.get("pptFileKey", "")
        ppt_file_name = idea_doc.get("pptFileName", "")
        ppt_file_url = idea_doc.get("pptFileUrl", "")
        ppt_file_size = idea_doc.get("pptFileSize", 0)
        
        print(f"   innovatorId: {innovator_id}")
        print(f"   title: {title}")
        print(f"   concept length: {len(concept)} chars")
        print(f"   background length: {len(background)} chars")
        print(f"   pptFileKey: {ppt_file_key or 'Not available'}")
        print(f"   pptFileName: {ppt_file_name or 'Not available'}")
        
        # Step 6: Determine validation source
        print("\n[STEP 6] 🔍 Determining validation source...")
        
        temp_path = None
        extracted_idea_name = title
        extracted_idea_concept = concept or background or "No detailed concept provided"
        
        # If PPT file exists, download and process it
        if ppt_file_url and ppt_file_key:
            print("✅ PPT file available - will download from S3")
            
            try:
                # Step 7: Download from S3
                print("\n[STEP 7] 📥 Downloading PPT from S3...")
                
                file_ext = os.path.splitext(ppt_file_name)[1].lower()
                if file_ext not in [".pdf", ".ppt", ".pptx"]:
                    raise Exception(f"Invalid file extension: {file_ext}")
                
                print(f"   File extension: {file_ext}")
                
                import tempfile
                import boto3
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_ext,
                    prefix="validate_pitch_",
                    mode='w+b'
                )
                temp_file.close()
                temp_path = temp_file.name
                
                print(f"   Temp file created: {temp_path}")
                
                try:
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                        region_name=os.getenv("AWS_REGION", "us-east-1")
                    )
                    
                    bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "pragati-docs")
                    
                    print(f"   Downloading from bucket: {bucket_name}")
                    print(f"   Key: {ppt_file_key}")
                    
                    s3_client.download_file(bucket_name, ppt_file_key, temp_path)
                    
                    print(f"✅ File downloaded successfully")
                    print(f"   File size: {os.path.getsize(temp_path)} bytes")
                    
                except Exception as s3_err:
                    print(f"❌ S3 download failed: {s3_err}")
                    raise Exception(f"S3 download failed: {str(s3_err)}")
                
                # Step 8: Process pitch deck
                print("\n[STEP 8] 📄 Processing pitch deck...")
                
                try:
                    extracted_info = pitch_deck_processor.process_pitch_deck(temp_path)
                    extracted_idea_name = extracted_info.get("idea_name", title)
                    extracted_idea_concept = extracted_info.get("idea_concept", concept)
                    
                    print(f"✅ Pitch deck processed")
                    print(f"   Extracted idea name: {extracted_idea_name}")
                    print(f"   Extracted concept length: {len(extracted_idea_concept)} chars")
                    
                except Exception as process_err:
                    print(f"⚠️ Pitch deck processing failed: {process_err}")
                    print(f"   Falling back to database concept/background")
                    extracted_idea_name = title
                    extracted_idea_concept = concept or background
                    
            except Exception as download_err:
                print(f"⚠️ Download/processing failed: {download_err}")
                print(f"   Using database concept/background instead")
                
        else:
            print("ℹ️ No PPT file - using database concept/background directly")
        
        # Step 9: Run AI validation
        print("\n[STEP 9] 🤖 Running AI validation...")
        print(f"   Idea name: {extracted_idea_name}")
        print(f"   Concept length: {len(extracted_idea_concept)} chars")
        print(f"   Custom weights: {custom_weights is not None}")
        
        try:
            broadcast_agent_message("System", f"🚀 Validating '{extracted_idea_name}'", "system")
        except:
            pass
        
        try:
            validation_result = validate_idea(
                extracted_idea_name,
                extracted_idea_concept,
                custom_weights
            )
            
            print("✅ Validation completed")
            print(f"   Result type: {type(validation_result)}")
            
        except Exception as val_err:
            print(f"❌ Validation failed: {val_err}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Validation failed: {str(val_err)}")
        
        if not validation_result or validation_result.get("error"):
            error_msg = validation_result.get("error", "Unknown error") if validation_result else "Validation returned null"
            print(f"❌ Validation error: {error_msg}")
            return jsonify({
                "error": "Validation failed",
                "details": error_msg
            }), 500
        
        # Step 10: Add personalized insights
        print("\n[STEP 10] 👤 Adding personalized insights...")
        
        try:
            profile_manager = get_user_profile_manager()
            user_context = profile_manager.get_personalized_validation_context(innovator_id)
            
            if user_context.get('has_profile'):
                print(f"✅ User profile found")
                validation_result['personalized_insights'] = {
                    "user_fit_score": user_context.get('fit_score'),
                    "entrepreneurial_fit": user_context.get('entrepreneurial_fit'),
                    "strengths_to_leverage": user_context.get('strengths', [])[:3],
                    "areas_to_focus": user_context.get('weak_areas', [])[:3]
                }
            else:
                print("ℹ️ No user profile found")
                
        except Exception as profile_err:
            print(f"⚠️ Profile fetch failed (non-critical): {profile_err}")
        
        # Step 11: Generate detailed analysis
        print("\n[STEP 11] 📊 Generating detailed analysis...")
        
        try:
            if db_manager:
                detailed_analysis = db_manager.generate_detailed_report_data(
                    validation_result,
                    extracted_idea_name,
                    extracted_idea_concept
                )
                validation_result['detailed_analysis'] = detailed_analysis
                print("✅ Detailed analysis generated")
        except Exception as detail_err:
            print(f"⚠️ Detailed analysis failed: {detail_err}")
            validation_result['detailed_analysis'] = {}
        
        # Step 12: Normalize result structure
        print("\n[STEP 12] 🔧 Normalizing result structure...")
        
        overall_score = validation_result.get("overall_score", 0)
        validation_outcome = validation_result.get("validation_outcome", "UNKNOWN")
        
        normalized_result = {
            "overall_score": overall_score,
            "validation_outcome": validation_outcome,
            "evaluated_data": validation_result.get("evaluated_data", {}),
            "action_points": validation_result.get("action_points", []),
            "total_agents_consulted": validation_result.get("total_agents_consulted", 0),
            "api_calls_made": validation_result.get("api_calls_made", 0),
            "consensus_level": validation_result.get("consensus_level", 0),
            "processing_time": validation_result.get("processing_time", 0),
            "detailed_viability_assessment": validation_result.get("detailed_viability_assessment", {}),
            "detailed_analysis": validation_result.get("detailed_analysis", {}),
            "roadmap": validation_result.get("roadmap", {}),
            "raw_validation_result": validation_result.get("raw_validation_result", {}),
            "ai_report": validation_result.get("ai_report", ""),
            "personalized_insights": validation_result.get("personalized_insights"),
            **{k: v for k, v in validation_result.items() if k not in [
                "overall_score", "validation_outcome", "evaluated_data",
                "action_points", "total_agents_consulted", "api_calls_made",
                "consensus_level", "processing_time", "detailed_viability_assessment",
                "detailed_analysis", "roadmap", "raw_validation_result", "ai_report",
                "personalized_insights"
            ]}
        }
        
        validation_result = normalized_result
        print("✅ Result normalized")
        
        # Step 13: Save to database
        print("\n[STEP 13] 💾 Saving to database...")
        
        report_id = None
        
        try:
            result_doc = {
                "ideaId": str(idea_id),
                "innovatorId": innovator_id,
                "title": title,
                "concept": concept,
                "background": background,
                "originalFilename": ppt_file_name,
                "pptFileKey": ppt_file_key,
                "pptFileSize": ppt_file_size,
                "source_type": "pitch_deck_single",
                "extractedIdeaName": extracted_idea_name,
                "extractedIdeaConcept": extracted_idea_concept,
                "validationResult": validation_result,
                "overallScore": overall_score,
                "validationOutcome": validation_outcome,
                "totalAgentsConsulted": validation_result.get("total_agents_consulted", 0),
                "apiCallsMade": validation_result.get("api_calls_made", 0),
                "consensusLevel": validation_result.get("consensus_level", 0),
                "processingTime": validation_result.get("processing_time", 0),
                "actionPoints": validation_result.get("action_points", []),
                "detailedViabilityAssessment": validation_result.get("detailed_viability_assessment", {}),
                "detailedAnalysis": validation_result.get("detailed_analysis", {}),
                "roadmap": validation_result.get("roadmap", {}),
                "rawValidationResult": validation_result.get("raw_validation_result", {}),
                "aiReport": validation_result.get("ai_report", ""),
                "evaluatedData": validation_result.get("evaluated_data", {}),
                "validationTimestamp": datetime.utcnow().isoformat(),
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }
            
            results_collection = db_manager.db['results']
            insert_result = results_collection.insert_one(result_doc)
            report_id = str(insert_result.inserted_id)
            
            print(f"✅ Result saved to database")
            print(f"   Report ID: {report_id}")
            
            # Update idea document
            ideas_collection.update_one(
                {"_id": idea_id},
                {"$set": {
                    "overallScore": overall_score,
                    "status": validation_outcome,
                    "resultDocId": report_id,
                    "validationTimestamp": datetime.utcnow().isoformat(),
                }}
            )
            
            print(f"✅ Idea document updated")
            
        except Exception as db_err:
            print(f"❌ Database save failed: {db_err}")
            import traceback
            traceback.print_exc()
        
        # Step 14: Add metadata to result
        print("\n[STEP 14] 📋 Adding metadata to result...")
        
        validation_result['ideaId'] = str(idea_id)
        validation_result['innovatorId'] = innovator_id
        validation_result['title'] = title
        validation_result['extractedIdeaName'] = extracted_idea_name
        validation_result['extractedIdeaConcept'] = extracted_idea_concept
        validation_result['originalFilename'] = ppt_file_name
        
        if report_id:
            validation_result['report_id'] = report_id
            validation_result['resultDocId'] = report_id
            validation_result['saved_to_database'] = True
            validation_result['savedToDatabase'] = True
        else:
            validation_result['saved_to_database'] = False
            validation_result['savedToDatabase'] = False
        
        validation_result['source_type'] = 'pitch_deck_single'
        validation_result['validationTimestamp'] = datetime.utcnow().isoformat()
        
        print("✅ Metadata added")
        
        # Step 15: Clean up temp file
        print("\n[STEP 15] 🧹 Cleaning up...")
        
        if temp_path:
            try:
                os.unlink(temp_path)
                print(f"✅ Temp file deleted: {temp_path}")
            except Exception as cleanup_err:
                print(f"⚠️ Cleanup failed: {cleanup_err}")
        
        # Step 16: Return response
        print("\n[STEP 16] 🎉 Returning response...")
        print(f"   Result keys: {list(validation_result.keys())}")
        
        try:
            broadcast_agent_message("System", f"✅ Validation complete! Score: {overall_score:.2f}", "success")
        except:
            pass
        
        print("="*80)
        print("✅ VALIDATE-PITCH-DECK COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
        return jsonify(validation_result), 200
        
    except Exception as e:
        print(f"\n" + "="*80)
        print(f"❌ FATAL ERROR IN VALIDATE-PITCH-DECK")
        print(f"="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        
        return jsonify({
            "error": "Failed to validate pitch deck",
            "details": str(e)
        }), 500


@app.route('/api/test-validation', methods=['GET'])
def test_validation():
    """Test endpoint for quick validation testing"""
    try:
        test_result = validate_idea(
            "Smart Agriculture IoT Platform",
            "An IoT-based platform that uses sensors and AI to help farmers monitor soil conditions, weather patterns, and crop health in real-time, with automated irrigation and fertilizer recommendations optimized for Indian farming conditions."
        )
        
        return jsonify({
            "success": True,
            "test_result": {
                "overall_score": test_result["overall_score"],
                "validation_outcome": test_result["validation_outcome"],
                "processing_time": test_result.get("processing_time", 0),
                "agents_consulted": test_result.get("api_calls_made", 0),
                "consensus_level": test_result.get("consensus_level", 0)
            },
            "message": "Test validation completed successfully"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Test validation failed",
            "details": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Serve the main web UI"""
    return send_from_directory('../static', 'index.html')

@app.route('/reports.html', methods=['GET'])
def reports_page():
    """Serve the reports viewer page"""
    return send_from_directory('../static', 'reports.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Pragati Backend v3.0",
        "description": "AI-powered idea validation using 109+ specialized CrewAI agents",
        "version": "3.0.0",
        "endpoints": {
            "POST /api/validate-idea": "Validate ideas using multi-agent system (JSON input)",
            "POST /api/validate-pitch-deck": "Validate ideas from PDF/PPT pitch deck upload",
            "GET /api/framework-info": "Get evaluation framework information",
            "GET /api/system-info": "Get system information",
            "GET /api/agents": "Get information about all validation agents",
            "GET /api/test-validation": "Test the validation system",
            "GET /api/reports/<user_id>": "Get all reports for a user (JSON)",
            "GET /api/report/<report_id>": "Get specific report by ID (JSON)",
            "GET /api/report/<report_id>/download": "Download report as PDF",
            "GET /report/<report_id>": "Get report data for UI display (JSON)",
            "POST /api/psychometric/generate": "Generate psychometric assessment (20 questions)",
            "POST /api/psychometric/evaluate": "Evaluate psychometric responses and get summary",
            "GET /api/psychometric/evaluations/<user_id>": "Get all psychometric evaluations for user",
            "GET /api/psychometric/evaluation/<evaluation_id>": "Get specific psychometric evaluation",
            "GET /api/profile/<user_id>": "Get user profile (from psychometric assessment)",
            "GET /api/profile/<user_id>/validation-context": "Get personalized validation context",
            "GET /": "Main validation interface (web UI)",
            "GET /reports.html": "Reports viewer page (web UI)",
            "GET /health": "Health check endpoint"
        },
        "features": [
            "109+ specialized AI validation agents",
            "Inter-agent collaboration and dependency resolution",
            "Comprehensive Indian market analysis",
            "Real-time consensus building",
            "Detailed PDF reporting with download",
            "Psychometric assessment for entrepreneurs",
            "User profiles with personalized validation insights",
            "Profile-aware recommendations",
            "Beautiful modern web interface"
        ]
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf_report():
    """Generate PDF report for validation result"""
    try:
        data = request.get_json()
        idea_name = data.get('idea_name', '')
        idea_concept = data.get('idea_concept', '')
        
        if not idea_name or not idea_concept:
            return jsonify({
                "error": "Both idea_name and idea_concept are required"
            }), 400
        
        # Run validation (synchronous method)
        result = validate_idea(idea_name, idea_concept)
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.pdf"
        output_path = os.path.join('../static/reports', filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate PDF
        pdf_path = pdf_generator.generate_report(result, idea_name, idea_concept, output_path)
        
        return jsonify({
            "success": True,
            "pdf_url": f"/static/reports/{filename}",
            "filename": filename,
            "validation_result": {
                "overall_score": result.get("overall_score", 0),
                "validation_outcome": result.get("validation_outcome", "UNKNOWN"),
                "total_agents_consulted": result.get("api_calls_made", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return jsonify({
            "error": f"PDF generation failed: {str(e)}"
        }), 500

@app.route('/api/download-pdf/<filename>')
def download_pdf(filename):
    """Download generated PDF report"""
    try:
        return send_file(f'../static/reports/{filename}', as_attachment=True)
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        return jsonify({
            "error": f"PDF download failed: {str(e)}"
        }), 404

@app.route('/api/agent-stream')
def agent_stream():
    """Server-Sent Events endpoint for real-time agent messages"""
    def event_stream():
        connection_id = id(threading.current_thread())
        active_connections.add(connection_id)
        
        try:
            while connection_id in active_connections:
                try:
                    # Get message from queue with timeout
                    message = agent_message_queue.get(timeout=1.0)
                    yield f"data: {message}\n\n"
                    agent_message_queue.task_done()
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    import json
                    heartbeat = json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})
                    yield f"data: {heartbeat}\n\n"
        except GeneratorExit:
            active_connections.discard(connection_id)
        finally:
            active_connections.discard(connection_id)
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})

@app.route('/api/test-message', methods=['POST'])
def test_message():
    """Test endpoint to send a message via SSE"""
    try:
        data = request.get_json()
        agent_name = data.get('agent', 'Test Agent')
        message = data.get('message', 'Test message')
        message_type = data.get('type', 'info')
        
        broadcast_agent_message(agent_name, message, message_type)
        
        return jsonify({
            "success": True,
            "message": "Test message sent"
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

def broadcast_agent_message(agent_name, message, message_type="info"):
    """Broadcast message to all connected clients"""
    import json
    
    # Clean the message to ensure valid JSON
    clean_message = str(message).replace('\n', ' ').replace('\r', ' ').strip()
    clean_agent = str(agent_name).replace('\n', ' ').replace('\r', ' ').strip()
    
    data = {
        'type': str(message_type),
        'agent': clean_agent,
        'message': clean_message,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        json_data = json.dumps(data, ensure_ascii=False)
        agent_message_queue.put(json_data, block=False)
        logger.debug(f"Broadcast message: {json_data}")
    except (queue.Full, TypeError, ValueError) as e:
        logger.warning(f"Failed to broadcast message: {e}")
        pass

# Make broadcast function globally available
import builtins
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking only
    pass
else:
    # At runtime, add to builtins
    builtins.broadcast_agent_message = broadcast_agent_message  # type: ignore


if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # ============================================================
    # DEBUG: Print all environment variables
    # ============================================================
    # print("\n" + "="*70)
    # print("🔍 ENVIRONMENT VARIABLES CHECK")
    # print("="*70)
    
    # Check OPENAI_API_KEY
    openai_key = os.getenv('OPENAI_API_KEY')
    # if openai_key:
    #     masked_key = openai_key[:10] + "..." + openai_key[-5:]
    #     print(f"✅ OPENAI_API_KEY: {masked_key}")
    # else:
    #     print(f"❌ OPENAI_API_KEY: NOT SET")
    
    # Check PORT
    port = int(os.getenv('PORT', 5001))
    # print(f"✅ PORT: {port}")
    
    # Check FLASK_ENV
    flask_env = os.getenv('FLASK_ENV', 'development')
    # print(f"✅ FLASK_ENV: {flask_env}")
    
    # Check MONGODB_URI (if exists)
    mongo_uri = os.getenv('MONGODB_URI')
    # if mongo_uri:
    #     print(f"✅ MONGODB_URI: {mongo_uri[:50]}...")
    # else:
    #     print(f"⚠️  MONGODB_URI: NOT SET (using default if configured)")
    
    # Check other important env vars
    # print("\n📋 Additional Environment Variables:")
    # for key in os.environ:
    #     if any(x in key.upper() for x in ['API', 'KEY', 'SECRET', 'TOKEN', 'DATABASE', 'DB']):
    #         value = os.getenv(key)
    #         if value and len(value) > 20:
    #             masked = value[:10] + f"...({len(value)-15} chars)..." + value[-5:]
    #             print(f"   {key}: {masked}")
    #         elif value:
    #             print(f"   {key}: {value}")
    
    # print("="*70 + "\n")
    
    # ============================================================
    # Validation
    # ============================================================
    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable is required!")
        print("\n📌 How to set it:")
        print("   Linux/Mac:   export OPENAI_API_KEY=sk-...")
        print("   Windows CMD: set OPENAI_API_KEY=sk-...")
        print("   Windows PS:  $env:OPENAI_API_KEY='sk-...'")
        print("   OR create .env file with: OPENAI_API_KEY=sk-...")
        exit(1)
    
    # print("✅ All required environment variables are set!\n")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Pragati Backend v3.0")
    logger.info("   Multi-Agent System with CrewAI")
    logger.info("=" * 60)
    
    # Initialize the validation system on startup
    try:
        from crew_ai_integration import get_pragati_validator
        print("🤖 Initializing CrewAI validation system...")
        validator = get_pragati_validator()
        logger.info(f"✅ Successfully initialized {len(validator.agents)} validation agents")
        print(f"✅ Initialized {len(validator.agents)} agents\n")
    except Exception as e:
        logger.error(f"❌ Failed to initialize validation system: {e}")
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Get configuration
    debug_mode = flask_env == 'development'
    host = '0.0.0.0'
    
    logger.info("=" * 60)
    logger.info(f"📍 Server Configuration:")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   URL: http://{host}:{port}")
    logger.info(f"   Debug: {debug_mode}")
    logger.info("=" * 60)
    
    # Print startup info
    # print("="*70)
    # print("📍 SERVER CONFIGURATION")
    # print("="*70)
    # print(f"   Host: {host}")
    # print(f"   Port: {port}")
    # print(f"   URL:  http://{host}:{port}")
    # print(f"   Debug Mode: {debug_mode}")
    # print(f"   CORS Enabled: Yes")
    # print("="*70)
    # print("\n🚀 Flask server is starting...\n")
    
    # Run the Flask app
    try:
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode
        )
    except KeyboardInterrupt:
        print("\n✋ Server stopped by user")
        logger.info("✋ Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
