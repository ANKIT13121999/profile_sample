from xyz import chat
from langchain.schema import Human, System
import os
import glob
from pathlib import Path

class DDLToSQLGenerator:
    def __init__(self):
        self.system_prompt = """You are an expert SQL database engineer specializing in Snowflake DDL analysis and SQL query generation.

Your task is to:
1. Analyze the provided Snowflake DDL statements
2. Identify changes between old and new DDL (added columns, modified columns, renamed columns)
3. Generate appropriate SQL queries for the changes

When generating SQL queries:
- For added columns: Generate ALTER TABLE ADD COLUMN statements
- For modified columns: Generate ALTER TABLE MODIFY COLUMN statements
- For renamed columns: Generate ALTER TABLE RENAME COLUMN statements
- For dropped columns: Generate ALTER TABLE DROP COLUMN statements
- Include proper data types, constraints, and default values
- Follow Snowflake SQL syntax standards

Return ONLY the SQL queries without explanations unless asked."""

    def generate_alter_statements(self, old_ddl, new_ddl):
        """Generate ALTER statements by comparing old and new DDL"""
        messages = [
            System(content=self.system_prompt),
            Human(content=f"""Analyze these DDL changes and generate the necessary ALTER TABLE statements:

OLD DDL:
{old_ddl}

NEW DDL:
{new_ddl}

Generate the SQL ALTER statements needed to transform the old DDL to the new DDL.""")
        ]
        
        response = chat(messages)
        return response
    
    def generate_queries_from_ddl(self, ddl, query_type):
        """Generate specific SQL queries based on DDL"""
        query_prompts = {
            "select": "Generate a comprehensive SELECT query that retrieves all columns from this table with proper formatting.",
            "insert": "Generate an INSERT statement template with all columns for this table.",
            "update": "Generate an UPDATE statement template that can update key columns in this table.",
            "merge": "Generate a MERGE statement template for upserting data into this table.",
            "analysis": "Generate analytical queries (aggregations, grouping) that would be useful for this table structure."
        }
        
        messages = [
            System(content=self.system_prompt),
            Human(content=f"""Based on this Snowflake DDL:

{ddl}

{query_prompts.get(query_type, query_prompts['select'])}""")
        ]
        
        response = chat(messages)
        return response
    
    def detect_and_generate_changes(self, old_ddl, new_ddl, table_name):
        """Detect specific changes and generate targeted SQL"""
        messages = [
            System(content=self.system_prompt),
            Human(content=f"""Compare these two DDL versions for table '{table_name}' and:

1. List all detected changes (added columns, modified columns, type changes, constraint changes)
2. Generate the exact ALTER TABLE statements needed
3. Add comments explaining each change

OLD DDL:
{old_ddl}

NEW DDL:
{new_ddl}

Format output as:
-- DETECTED CHANGES
-- [list changes here]

-- GENERATED SQL
[SQL statements here]""")
        ]
        
        response = chat(messages)
        return response
    
    def generate_analytical_query(self, updated_ddl, user_question):
        """Generate analytical SQL query based on user's question and updated DDL"""
        analytical_prompt = """You are an expert SQL analyst specializing in Snowflake.

Your task is to:
1. Understand the table structure from the provided DDL
2. Interpret the user's analytical question
3. Generate an optimized Snowflake SQL query that answers their question

Guidelines:
- Use appropriate aggregations (SUM, AVG, COUNT, MIN, MAX)
- Include GROUP BY, HAVING, ORDER BY as needed
- Use window functions when appropriate (ROW_NUMBER, RANK, LAG, LEAD)
- Add CTEs (WITH clauses) for complex queries
- Include proper date/time functions for temporal analysis
- Add comments explaining the query logic
- Optimize for Snowflake performance (use appropriate JOIN types, filtering)
- Format the query for readability

Return ONLY the SQL query with inline comments."""

        messages = [
            System(content=analytical_prompt),
            Human(content=f"""Table DDL:
{updated_ddl}

User Question: {user_question}

Generate an analytical SQL query that answers this question.""")
        ]
        
        response = chat(messages)
        return response


# Example Usage
if __name__ == "__main__":
    generator = DDLToSQLGenerator()
    
    # Example 1: Detecting changes between DDLs
    old_ddl = """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        email VARCHAR(100)
    );
    """
    
    new_ddl = """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        email VARCHAR(100),
        phone VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'active'
    );
    """
    
    print("=== Example 1: Generate ALTER statements ===")
    result1 = generator.generate_alter_statements(old_ddl, new_ddl)
    print(result1)
    
    # Example 2: Generate specific query types
    print("\n=== Example 2: Generate INSERT template ===")
    result2 = generator.generate_queries_from_ddl(new_ddl, "insert")
    print(result2)
    
    # Example 3: Detailed change detection
    print("\n=== Example 3: Detailed change analysis ===")
    result3 = generator.detect_and_generate_changes(old_ddl, new_ddl, "customers")
    print(result3)
    
    # Example 4: Generate analytical queries
    print("\n=== Example 4: Analytical queries ===")
    result4 = generator.generate_queries_from_ddl(new_ddl, "analysis")
    print(result4)
    
    # Example 5: Generate analytical query based on user question
    print("\n=== Example 5: User Question-based Analytical Query ===")
    user_question = "Show me the top 5 customers by email domain who were created in the last month and are active"
    result5 = generator.generate_analytical_query(new_ddl, user_question)
    print(result5)
    
    # Example 6: More complex analytical questions
    print("\n=== Example 6: Complex Analytical Query ===")
    complex_question = "What is the month-over-month growth rate of new active customers, grouped by their status?"
    result6 = generator.generate_analytical_query(new_ddl, complex_question)
    print(result6)
    
    # Example 7: Analytical query with aggregations
    print("\n=== Example 7: Aggregation Query ===")
    agg_question = "Calculate the total count of customers by status and show the percentage distribution"
    result7 = generator.generate_analytical_query(new_ddl, agg_question)
    print(result7)
