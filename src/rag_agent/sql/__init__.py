"""Safe Text-to-SQL pipeline over the Olist analytics database (data/olist.db).

Flow: question -> generate_sql() -> validate_sql() -> execute_sql() -> answer.
See text_to_sql.py for the orchestrator, validator.py for the security gate.
"""
