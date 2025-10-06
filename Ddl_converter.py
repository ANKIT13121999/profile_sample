from xyz import chat
from langchain.schema import Human, System

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


# Advanced Example: Batch Processing Multiple Tables
class BatchDDLProcessor:
    def __init__(self):
        self.generator = DDLToSQLGenerator()
    
    def process_schema_changes(self, schema_changes):
        """
        Process multiple table changes
        schema_changes: dict with table names as keys and (old_ddl, new_ddl) tuples as values
        """
        results = {}
        
        for table_name, (old_ddl, new_ddl) in schema_changes.items():
            print(f"\nProcessing table: {table_name}")
            results[table_name] = self.generator.detect_and_generate_changes(
                old_ddl, new_ddl, table_name
            )
        
        return results
    
    def generate_migration_script(self, schema_changes):
        """Generate a complete migration script for all changes"""
        messages = [
            System(content=self.generator.system_prompt),
            Human(content=f"""Generate a complete Snowflake migration script for these schema changes:

{schema_changes}

Include:
1. Transaction control (BEGIN/COMMIT/ROLLBACK)
2. All ALTER statements in correct order
3. Comments for each change
4. Verification queries to check changes
5. Rollback statements in comments

Format as a production-ready migration script.""")
        ]
        
        response = chat(messages)
        return response


# Usage with custom prompt engineering
def custom_sql_generation(ddl, custom_requirements):
    """Allow custom requirements for SQL generation"""
    system_prompt = """You are an expert SQL engineer. Generate SQL based on DDL and specific requirements."""
    
    messages = [
        System(content=system_prompt),
        Human(content=f"""DDL:
{ddl}

Requirements:
{custom_requirements}

Generate the appropriate SQL queries.""")
    ]
    
    response = chat(messages)
    return response


# Example with specific business logic
ddl = """
CREATE TABLE orders (
    order_id INT,
    customer_id INT,
    order_date TIMESTAMP,
    total_amount DECIMAL(10,2),
    status VARCHAR(20)
);
"""

requirements = """
1. Create a query to find top 10 customers by total order value
2. Create a query to find orders placed in the last 30 days
3. Create a query to calculate monthly revenue
4. All queries should include proper date formatting and number formatting
"""

result = custom_sql_generation(ddl, requirements)
print("\n=== Custom Business Logic Queries ===")
print(result)
