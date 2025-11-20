"""
Report Management Endpoints for Pragati AI Engine
Returns structured JSON data - UI will handle presentation
"""

import logging
from flask import jsonify, request, send_file, Response, stream_with_context, render_template, send_from_directory
from database_manager import get_database_manager
from pdf_report_system import generate_validation_report
from pdf_report_system.report_writer import AIReportWriter
from pdf_report_system.data_processor import AgentDataProcessor
from datetime import datetime
import json
import queue
import threading
from io import BytesIO
import os

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


def register_report_endpoints(app):
    """Register report management endpoints with Flask app"""
    
    @app.route('/api/reports/<user_id>', methods=['GET'])
    def get_user_reports(user_id):
        """Get all reports for a specific user"""
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            limit = request.args.get('limit', 10, type=int)
            reports = db_manager.get_user_reports(user_id, limit)
            
            return jsonify({
                "user_id": user_id,
                "reports": reports,
                "count": len(reports)
            })
            
        except Exception as e:
            logger.error(f"Failed to get user reports: {e}")
            return jsonify({
                "error": "Failed to retrieve reports",
                "details": str(e)
            }), 500

    @app.route('/api/report/<report_id>', methods=['GET'])
    def get_report_data(report_id):
        """Get specific report data by ID (full detailed analysis)"""
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            return jsonify(report)
            
        except Exception as e:
            logger.error(f"Failed to get report: {e}")
            return jsonify({
                "error": "Failed to retrieve report",
                "details": str(e)
            }), 500

    @app.route('/report/<report_id>', methods=['GET'])
    def get_report_for_display(report_id):
        """
        Serve the dedicated report view page
        Returns HTML page that renders report content from agent conversations
        """
        try:
            # Verify report exists
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Serve the HTML report page
            static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
            return send_from_directory(static_dir, 'report_view.html')
            
        except Exception as e:
            logger.error(f"Failed to serve report page: {e}")
            return jsonify({
                "error": "Failed to load report page",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/generate', methods=['GET'])
    def generate_ai_report(report_id):

        logger.info(color_log("=" * 80, LogColors.BG_CYAN))
        logger.info(color_log("🔍 DEBUG: About to save to database", LogColors.BG_YELLOW))
        """
        Generate AI-written comprehensive report from agent conversations
        Uses gpt-4.1-mini to create detailed bullet-pointed report
        Caches the result in MongoDB - only generates on first request
        
        ✅ NOW SAVES detailed_viability_assessment TO DATABASE
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            # Check if cached AI report exists
            cached_report = db_manager.get_ai_report(report_id)
            if cached_report:
                logger.info(f"✅ Returning cached AI report for {report_id}")
                return jsonify({
                    "success": True,
                    "report_id": report_id,
                    "ai_report": cached_report["ai_report"],
                    "cached": True,
                    "generated_at": cached_report["generated_at"].isoformat() if cached_report.get("generated_at") else None
                })
            
            # No cached report - need to generate
            report = db_manager.get_report_by_id(report_id)
            
            # 🔍 =================================================
            # 🔍 DEBUG: Inspect the raw report from the database
            # 🔍 =================================================
            if report:
                logger.info(color_log("=" * 80, LogColors.BG_BLUE))
                logger.info(color_log("🔍 DEBUG: Raw report data from MongoDB", LogColors.BG_YELLOW))
                logger.info(f"   Report ID: {report.get('_id')}")
                logger.info(f"   Top-level keys: {list(report.keys())}")

                # Check for expected data containers
                if 'evaluated_data' in report:
                    logger.info(f"   ✅ 'evaluated_data' key exists. Type: {type(report['evaluated_data']).__name__}")
                    if isinstance(report['evaluated_data'], dict):
                        logger.info(f"      First 3 cluster keys: {list(report['evaluated_data'].keys())[:3]}")
                else:
                    logger.warning("   ❌ 'evaluated_data' key NOT FOUND at top level.")

                if 'raw_validation_result' in report:
                    logger.info(f"   ✅ 'raw_validation_result' key exists.")
                    if isinstance(report['raw_validation_result'], dict) and 'evaluated_data' in report['raw_validation_result']:
                        logger.info("      ✅ 'evaluated_data' FOUND inside 'raw_validation_result'.")
                    else:
                        logger.warning("      ❌ 'evaluated_data' NOT FOUND inside 'raw_validation_result'.")
                
                if 'detailed_analysis' in report:
                    logger.info(f"   ✅ 'detailed_analysis' key exists.")
                    if isinstance(report['detailed_analysis'], dict) and 'cluster_analyses' in report['detailed_analysis']:
                        logger.info("      ✅ 'cluster_analyses' FOUND inside 'detailed_analysis'.")
                    else:
                        logger.warning("      ❌ 'cluster_analyses' NOT FOUND inside 'detailed_analysis'.")
                logger.info(color_log("=" * 80, LogColors.BG_BLUE))
            # 🔍 =================================================
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Extract agent conversations
            processor = AgentDataProcessor(report)
            processed_data = processor.process_complete_report_data()
            
            if not processed_data or not processed_data.get('all_conversations'):
                # Debug: Check what data is available
                has_evaluated_data = bool(report.get('evaluated_data'))
                has_raw_result = bool(report.get('raw_validation_result'))
                has_raw_evaluated = bool(report.get('raw_validation_result', {}).get('evaluated_data'))
                
                logger.warning(f"Report {report_id} - evaluated_data: {has_evaluated_data}, raw_result: {has_raw_result}, raw_evaluated: {has_raw_evaluated}")
                
                return jsonify({
                    "error": "No agent conversations found in this report",
                    "message": "This report does not contain agent conversation data. Please run a new validation to generate conversations.",
                    "debug": {
                        "has_evaluated_data": has_evaluated_data,
                        "has_raw_validation_result": has_raw_result,
                        "has_raw_evaluated_data": has_raw_evaluated
                    }
                }), 404
            
            # Generate AI-written report using gpt-4.1-mini
            logger.info(f"🔄 Generating new AI report for {report_id} using {len(processed_data['all_conversations'])} conversations")
            
            writer = AIReportWriter(progress_callback=None)
            ai_report = writer.write_comprehensive_report(
                processed_data['all_conversations'],
                processed_data['metadata']
            )
            
            # 🔍 DEBUG: Check what write_comprehensive_report actually returned
            logger.info(f"🔍 DEBUG: ai_report type = {type(ai_report)}")
            logger.info(f"🔍 DEBUG: ai_report keys = {list(ai_report.keys()) if isinstance(ai_report, dict) else 'NOT A DICT'}")
            
            # Check if detailed_viability_assessment exists
            if 'detailed_viability_assessment' in ai_report:
                dva = ai_report['detailed_viability_assessment']
                logger.info(f"🔍 DEBUG: detailed_viability_assessment type = {type(dva)}")
                logger.info(f"🔍 DEBUG: detailed_viability_assessment empty? {len(dva) == 0}")
                if isinstance(dva, dict):
                    logger.info(f"🔍 DEBUG: detailed_viability_assessment keys = {list(dva.keys())}")
                else:
                    logger.warning(f"🔍 WARNING: detailed_viability_assessment is {type(dva).__name__}, not dict!")
            else:
                logger.error("🔍 ERROR: 'detailed_viability_assessment' KEY NOT FOUND in ai_report!")
                logger.error(f"🔍 Available keys: {list(ai_report.keys())}")
            
            # ✅ ===== NEW BLOCK: EXTRACT AND SAVE TO DATABASE =====
            # Step 1: Extract detailed_viability_assessment for database storage
            logger.info("📊 Extracting detailed_viability_assessment from AI report...")
            detailed_viability_assessment = ai_report.get('detailed_viability_assessment', {})
            logger.info(f"✅ Extracted {len(detailed_viability_assessment)} clusters for database")
            
            # Step 2: Prepare data dictionary for database
            report_data_dict = {
                "success": True,
                "detailed_viability_assessment": detailed_viability_assessment,
                "ai_report": ai_report,
                "processed_data": processed_data,
                "metadata": processed_data.get('metadata', {})
            }
            
            # Step 3: Save to database
            logger.info(f"💾 Saving detailed_viability_assessment to database...")
            try:
                saved_report_id = db_manager.save_validation_report(
                    user_id=report.get('user_id', 'unknown'),
                    title=report.get('title', 'AI Generated Report'),
                    validation_result=report.get('raw_validation_result', {}),  # Keep original validation
                    idea_name=report.get('idea_name', 'Unknown'),
                    idea_concept=report.get('idea_concept', ''),
                    source_type=report.get('source_type', 'manual'),
                    detailed_viability_assessment=detailed_viability_assessment,  # ✅ Pass as separate param
                    ai_report=ai_report  # ✅ Also pass ai_report separately
                )
                logger.info(f"✅ Report saved/updated to database: {saved_report_id}")
            except Exception as db_error:
                logger.error(f"❌ Failed to save to database: {db_error}")
            # ✅ ===== END OF NEW BLOCK =====
            
            # Save to MongoDB for caching
            save_success = db_manager.save_ai_report(report_id, ai_report)
            if save_success:
                logger.info(f"✅ AI report generated and cached for {report_id}")
            else:
                logger.warning(f"⚠️ AI report generated but failed to cache for {report_id}")
            
            return jsonify({
                "success": True,
                "report_id": report_id,
                "ai_report": ai_report,
                "conversations_used": len(processed_data['all_conversations']),
                "cached": False,
                "saved_to_db": save_success,
                "detailed_viability_assessment": detailed_viability_assessment  # ← Add this to response
            })
            
        except Exception as e:
            logger.error(f"Failed to generate AI report: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": "Failed to generate AI report",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/download', methods=['GET'])
    def download_report_pdf(report_id):
        """
        Download report as PDF with progress streaming
        Uses Server-Sent Events (SSE) to stream progress updates
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Create a queue for progress updates
            progress_queue = queue.Queue()
            pdf_result = {'buffer': None, 'error': None, 'filename': None}
            
            def progress_callback(message: str, progress: float):
                """Callback to send progress updates"""
                try:
                    progress_queue.put({
                        'message': message,
                        'progress': progress
                    })
                except Exception as e:
                    logger.error(f"Error sending progress: {e}")
            
            def generate_pdf():
                """Generate PDF in background thread"""
                try:
                    # Generate PDF with progress callback
                    pdf_buffer = generate_validation_report(report, progress_callback=progress_callback)
                    
                    # Create filename
                    title = report.get('title', 'report').replace(' ', '_')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"validation_report_{title}_{timestamp}.pdf"
                    
                    pdf_result['buffer'] = pdf_buffer
                    pdf_result['filename'] = filename
                    progress_queue.put({'type': 'complete'})
                except Exception as e:
                    logger.error(f"Failed to generate PDF: {e}")
                    pdf_result['error'] = str(e)
                    progress_queue.put({'type': 'error', 'error': str(e)})
            
            # Start PDF generation in background thread
            pdf_thread = threading.Thread(target=generate_pdf, daemon=True)
            pdf_thread.start()
            
            def generate():
                """Generator function for SSE streaming"""
                try:
                    while True:
                        try:
                            # Get progress update (with timeout)
                            item = progress_queue.get(timeout=1)
                            
                            if item.get('type') == 'complete':
                                # PDF generation complete, send final message
                                yield f"data: {json.dumps({'message': 'PDF ready!', 'progress': 100, 'complete': True})}\n\n"
                                break
                            elif item.get('type') == 'error':
                                # Error occurred
                                error_msg = item.get('error', 'Unknown error')
                                yield f"data: {json.dumps({'message': f'Error: {error_msg}', 'progress': 0, 'error': True})}\n\n"
                                break
                            else:
                                # Progress update
                                yield f"data: {json.dumps({'message': item.get('message', ''), 'progress': item.get('progress', 0)})}\n\n"
                                
                        except queue.Empty:
                            # Send keepalive
                            yield f"data: {json.dumps({'message': 'Processing...', 'progress': -1})}\n\n"
                            
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {json.dumps({'message': f'Error: {str(e)}', 'progress': 0, 'error': True})}\n\n"
            
            # Return SSE response
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to start PDF generation: {e}")
            return jsonify({
                "error": "Failed to start PDF generation",
                "details": str(e)
            }), 500
    
    @app.route('/api/report/<report_id>/download-pdf-file', methods=['GET'])
    def download_pdf_file(report_id):
        """
        Actually download the generated PDF file
        Called after progress streaming is complete
        """
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return jsonify({
                    "error": "Database not available"
                }), 503
            
            report = db_manager.get_report_by_id(report_id)
            
            if not report:
                return jsonify({
                    "error": "Report not found"
                }), 404
            
            # Generate PDF (without progress for direct download)
            pdf_buffer = generate_validation_report(report)
            
            # Create filename
            title = report.get('title', 'report').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"validation_report_{title}_{timestamp}.pdf"
            
            # Send PDF file
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return jsonify({
                "error": "Failed to generate PDF",
                "details": str(e)
            }), 500
