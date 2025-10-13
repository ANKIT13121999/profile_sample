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
    
    def load_ddl_from_file(self, file_path):
        """Load DDL content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def process_folder_ddls(self, old_ddl_folder, new_ddl_folder, file_pattern="*.sql"):
        """
        Process multiple DDL files from two folders (old and new versions)
        
        Args:
            old_ddl_folder: Path to folder containing old DDL files
            new_ddl_folder: Path to folder containing new DDL files
            file_pattern: Pattern to match DDL files (default: *.sql)
        
        Returns:
            Dictionary with results for each table
        """
        results = {}
        
        # Get all DDL files from old folder
        old_files = glob.glob(os.path.join(old_ddl_folder, file_pattern))
        
        for old_file in old_files:
            filename = os.path.basename(old_file)
            table_name = os.path.splitext(filename)[0]
            
            # Construct corresponding new file path
            new_file = os.path.join(new_ddl_folder, filename)
            
            # Check if new file exists
            if not os.path.exists(new_file):
                print(f"Warning: No matching new DDL found for {filename}")
                results[table_name] = {
                    'status': 'skipped',
                    'reason': 'No new DDL file found'
                }
                continue
            
            # Load DDLs
            old_ddl = self.load_ddl_from_file(old_file)
            new_ddl = self.load_ddl_from_file(new_file)
            
            if old_ddl is None or new_ddl is None:
                results[table_name] = {
                    'status': 'error',
                    'reason': 'Failed to load DDL files'
                }
                continue
            
            # Generate ALTER statements
            print(f"\nProcessing: {table_name}")
            alter_statements = self.detect_and_generate_changes(old_ddl, new_ddl, table_name)
            
            results[table_name] = {
                'status': 'success',
                'old_file': old_file,
                'new_file': new_file,
                'alter_statements': alter_statements
            }
        
        return results
    
    def process_single_folder_ddls(self, ddl_folder, file_pattern="*.sql"):
        """
        Process multiple DDL files from a single folder (only new DDLs)
        Useful when you only have updated DDLs without old versions
        
        Args:
            ddl_folder: Path to folder containing DDL files
            file_pattern: Pattern to match DDL files (default: *.sql)
        
        Returns:
            Dictionary with DDL content for each table
        """
        results = {}
        
        # Get all DDL files
        ddl_files = glob.glob(os.path.join(ddl_folder, file_pattern))
        
        for ddl_file in ddl_files:
            filename = os.path.basename(ddl_file)
            table_name = os.path.splitext(filename)[0]
            
            # Load DDL
            ddl_content = self.load_ddl_from_file(ddl_file)
            
            if ddl_content is None:
                results[table_name] = {
                    'status': 'error',
                    'reason': 'Failed to load DDL file'
                }
                continue
            
            results[table_name] = {
                'status': 'success',
                'file': ddl_file,
                'ddl': ddl_content
            }
        
        return results
    
    def generate_analytical_queries_for_multiple_tables(self, ddl_folder, user_question, file_pattern="*.sql"):
        """
        Generate analytical queries across multiple tables based on user question
        
        Args:
            ddl_folder: Path to folder containing DDL files
            user_question: User's analytical question
            file_pattern: Pattern to match DDL files
        
        Returns:
            SQL query that may involve multiple tables
        """
        # Load all DDLs
        ddls = self.process_single_folder_ddls(ddl_folder, file_pattern)
        
        # Combine all DDLs into context
        combined_ddl = ""
        table_list = []
        for table_name, info in ddls.items():
            if info['status'] == 'success':
                combined_ddl += f"\n-- Table: {table_name}\n{info['ddl']}\n"
                table_list.append(table_name)
        
        # Generate multi-table analytical query
        analytical_prompt = """You are an expert SQL analyst specializing in Snowflake.

Your task is to:
1. Understand all table structures from the provided DDLs
2. Interpret the user's analytical question
3. Determine which tables are needed to answer the question
4. Generate an optimized Snowflake SQL query (may involve JOINs across multiple tables)

Guidelines:
- Use appropriate JOINs when multiple tables are needed
- Use aggregations (SUM, AVG, COUNT, MIN, MAX) as needed
- Include GROUP BY, HAVING, ORDER BY as needed
- Use window functions when appropriate (ROW_NUMBER, RANK, LAG, LEAD)
- Add CTEs (WITH clauses) for complex queries
- Include proper date/time functions for temporal analysis
- Add comments explaining the query logic and which tables are being used
- Optimize for Snowflake performance
- Format the query for readability

Return ONLY the SQL query with inline comments."""

        messages = [
            System(content=analytical_prompt),
            Human(content=f"""Available Tables and their DDLs:
{combined_ddl}

Available Tables: {', '.join(table_list)}

User Question: {user_question}

Generate an analytical SQL query that answers this question using the appropriate tables.""")
        ]
        
        response = chat(messages)
        return response
    
    def save_results_to_file(self, results, output_file):
        """Save generated SQL results to a file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("-- Generated SQL Statements\n")
                f.write(f"-- Generated on: {os.popen('date').read().strip()}\n\n")
                
                for table_name, info in results.items():
                    f.write(f"\n{'='*80}\n")
                    f.write(f"-- Table: {table_name}\n")
                    f.write(f"-- Status: {info['status']}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    if info['status'] == 'success' and 'alter_statements' in info:
                        f.write(info['alter_statements'])
                        f.write("\n\n")
                    elif info['status'] != 'success':
                        f.write(f"-- {info.get('reason', 'Unknown error')}\n\n")
            
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


# Example Usage
if __name__ == "__main__":
    generator = DDLToSQLGenerator()
    
    print("="*80)
    print("DDL TO SQL GENERATOR - MULTIPLE DDL FILES SUPPORT")
    print("="*80)
    
    # ========================================================================
    # MAIN FEATURE 1: Process Multiple DDLs from Folders (Old vs New)
    # ========================================================================
    print("\n" + "="*80)
    print("FEATURE 1: BATCH PROCESS MULTIPLE DDL FILES")
    print("="*80)
    print("\nThis processes ALL .sql files in your folders automatically!")
    print("\nFolder Structure:")
    print("  ./ddls/old/")
    print("    ├── customers.sql")
    print("    ├── orders.sql")
    print("    ├── products.sql")
    print("    └── payments.sql")
    print("  ./ddls/new/")
    print("    ├── customers.sql")
    print("    ├── orders.sql")
    print("    ├── products.sql")
    print("    └── payments.sql")
    
    # Process all DDL files at once
    results = generator.process_folder_ddls(
        old_ddl_folder="./ddls/old",
        new_ddl_folder="./ddls/new",
        file_pattern="*.sql"  # Can be *.ddl or any pattern
    )
    
    # Display results for all tables
    print(f"\nProcessed {len(results)} tables:")
    for table_name, info in results.items():
        print(f"\n{'─'*80}")
        print(f"TABLE: {table_name}")
        print(f"Status: {info['status']}")
        if info['status'] == 'success':
            print(f"Old DDL: {info['old_file']}")
            print(f"New DDL: {info['new_file']}")
            print("\nGenerated ALTER Statements:")
            print(info['alter_statements'])
        else:
            print(f"Reason: {info.get('reason', 'Unknown')}")
    
    # Save all results to one migration script file
    generator.save_results_to_file(results, "./output/all_tables_migration.sql")
    print("\n✓ All ALTER statements saved to: ./output/all_tables_migration.sql")
    
    
    # ========================================================================
    # MAIN FEATURE 2: Multi-Table Analytical Queries
    # ========================================================================
    print("\n\n" + "="*80)
    print("FEATURE 2: ANALYTICAL QUERIES ACROSS MULTIPLE TABLES")
    print("="*80)
    print("\nThis can answer questions involving multiple tables with JOINs!")
    
    # Example: User asks a question that needs data from multiple tables
    ddl_folder = "./ddls/updated"
    
    # Question 1: Requires JOIN between customers and orders
    question1 = "Show me the top 10 customers by total order value with their email and phone"
    print(f"\n\nQuestion 1: {question1}")
    print("(This will automatically JOIN customers and orders tables)")
    
    query1 = generator.generate_analytical_queries_for_multiple_tables(
        ddl_folder=ddl_folder,
        user_question=question1
    )
    print("\nGenerated SQL:")
    print(query1)
    
    
    # Question 2: Requires multiple tables
    question2 = "Calculate monthly revenue by product category with customer status breakdown"
    print(f"\n\n{'-'*80}")
    print(f"Question 2: {question2}")
    print("(This will use orders, products, and customers tables)")
    
    query2 = generator.generate_analytical_queries_for_multiple_tables(
        ddl_folder=ddl_folder,
        user_question=question2
    )
    print("\nGenerated SQL:")
    print(query2)
    
    
    # Question 3: Complex aggregation
    question3 = "What is the average order value by customer status for customers created in the last 90 days?"
    print(f"\n\n{'-'*80}")
    print(f"Question 3: {question3}")
    
    query3 = generator.generate_analytical_queries_for_multiple_tables(
        ddl_folder=ddl_folder,
        user_question=question3
    )
    print("\nGenerated SQL:")
    print(query3)
    
    
    # ========================================================================
    # FEATURE 3: Process Single Folder (Only Updated DDLs)
    # ========================================================================
    print("\n\n" + "="*80)
    print("FEATURE 3: LOAD ALL DDLS FROM SINGLE FOLDER")
    print("="*80)
    print("\nUseful when you only have the latest DDL versions")
    
    ddl_results = generator.process_single_folder_ddls(
        ddl_folder="./ddls/updated",
        file_pattern="*.sql"
    )
    
    print(f"\nLoaded {len(ddl_results)} DDL files:")
    for table_name, info in ddl_results.items():
        if info['status'] == 'success':
            print(f"  ✓ {table_name:20s} - {info['file']}")
        else:
            print(f"  ✗ {table_name:20s} - {info.get('reason', 'Error')}")
    
    
    # ========================================================================
    # FEATURE 4: Single Table Analytical Query
    # ========================================================================
    print("\n\n" + "="*80)
    print("FEATURE 4: ANALYTICAL QUERY FOR SINGLE TABLE")
    print("="*80)
    
    # Get a specific table's DDL from the loaded results
    if 'customers' in ddl_results and ddl_results['customers']['status'] == 'success':
        customer_ddl = ddl_results['customers']['ddl']
        
        single_question = "Show me customers grouped by status with count and percentage, ordered by count descending"
        print(f"\nTable: customers")
        print(f"Question: {single_question}")
        
        single_query = generator.generate_analytical_query(customer_ddl, single_question)
        print("\nGenerated SQL:")
        print(single_query)
    
    
    # ========================================================================
    # ORIGINAL EXAMPLES: Basic Single DDL Operations
    # ========================================================================
    print("\n\n" + "="*80)
    print("BASIC EXAMPLES: Single DDL Processing (Original Functionality)")
    print("="*80)
    
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
