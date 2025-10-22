from xyz import chat
from langchain.schema import Human, System
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from inter.core.clients.xorclient import xorClient
import json
from datetime import datetime

class DDLVectorStore:
    """Manages DDL storage and retrieval using Milvus"""
    
    def __init__(self, host="localhost", port="19530", collection_name="ddl_store", 
                 embedding_model_name="bembedd-1rg", embedding_dim=1024):
        self.collection_name = collection_name
        
        # Initialize custom xorClient for embeddings
        self.embedding_client = xorClient()
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        
        # Create collection if not exists
        self._create_collection()
    
    def _create_collection(self):
        """Create Milvus collection for DDL storage"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="schema_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="ddl_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ddl_summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="column_names", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        
        schema = CollectionSchema(fields=fields, description="DDL Storage")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    
    def _extract_ddl_metadata(self, ddl_content):
        """Extract metadata from DDL for better searchability"""
        lines = ddl_content.strip().split('\n')
        
        # Extract table name
        table_name = ""
        for line in lines:
            if 'CREATE TABLE' in line.upper():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.upper() == 'TABLE' and i + 1 < len(parts):
                        table_name = parts[i + 1].replace('(', '').replace(';', '')
                        break
        
        # Extract column names
        columns = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('CREATE') and not line.startswith(')') and '(' in ddl_content:
                col_name = line.split()[0].replace(',', '').replace('(', '')
                if col_name and col_name.upper() not in ['PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT']:
                    columns.append(col_name)
        
        # Create summary
        summary = f"Table: {table_name}, Columns: {', '.join(columns)}"
        
        return table_name, columns, summary
    
    def _create_embedding(self, text):
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_ddl(self, ddl_content, schema_name="public"):
        """Store DDL in Milvus"""
        table_name, columns, summary = self._extract_ddl_metadata(ddl_content)
        
        # Create embedding from DDL content and summary
        embedding_text = f"{table_name} {summary} {ddl_content}"
        embedding = self._create_embedding(embedding_text)
        
        # Prepare data
        data = [{
            "table_name": table_name,
            "schema_name": schema_name,
            "ddl_content": ddl_content,
            "ddl_summary": summary,
            "column_names": json.dumps(columns),
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding
        }]
        
        # Insert into Milvus
        self.collection.insert(data)
        self.collection.flush()
        
        print(f"✓ Stored DDL for table: {table_name}")
        return table_name
    
    def search_similar_ddls(self, query, top_k=3):
        """Search for similar DDLs based on query"""
        # Generate query embedding
        query_embedding = self._create_embedding(query)
        
        # Load collection
        self.collection.load()
        
        # Search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["table_name", "schema_name", "ddl_content", "ddl_summary", "column_names"]
        )
        
        # Format results
        ddl_results = []
        for hits in results:
            for hit in hits:
                ddl_results.append({
                    "table_name": hit.entity.get("table_name"),
                    "schema_name": hit.entity.get("schema_name"),
                    "ddl_content": hit.entity.get("ddl_content"),
                    "ddl_summary": hit.entity.get("ddl_summary"),
                    "column_names": json.loads(hit.entity.get("column_names")),
                    "distance": hit.distance
                })
        
        return ddl_results
    
    def get_ddl_by_table_name(self, table_name):
        """Retrieve DDL by exact table name"""
        self.collection.load()
        
        expr = f'table_name == "{table_name}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["table_name", "schema_name", "ddl_content", "ddl_summary", "column_names"]
        )
        
        if results:
            return {
                "table_name": results[0].get("table_name"),
                "schema_name": results[0].get("schema_name"),
                "ddl_content": results[0].get("ddl_content"),
                "ddl_summary": results[0].get("ddl_summary"),
                "column_names": json.loads(results[0].get("column_names"))
            }
        return None
    
    def delete_ddl(self, table_name):
        """Delete DDL by table name"""
        expr = f'table_name == "{table_name}"'
        self.collection.delete(expr)
        print(f"✓ Deleted DDL for table: {table_name}")


class DDLToSQLGenerator:
    """Generate SQL queries using LLM with Milvus-backed DDL retrieval"""
    
    def __init__(self, milvus_host="localhost", milvus_port="19530"):
        self.vector_store = DDLVectorStore(host=milvus_host, port=milvus_port)
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
    
    def store_ddl(self, ddl_content, schema_name="public"):
        """Store DDL in vector database"""
        return self.vector_store.store_ddl(ddl_content, schema_name)
    
    def generate_alter_statements(self, new_ddl, table_name=None):
        """
        Generate ALTER statements by comparing new DDL with stored old DDL
        Uses similarity search to find the old DDL
        """
        # If table_name provided, get exact match; otherwise use similarity search
        if table_name:
            old_ddl_data = self.vector_store.get_ddl_by_table_name(table_name)
        else:
            # Extract table name from new DDL for search
            _, _, summary = self.vector_store._extract_ddl_metadata(new_ddl)
            similar_ddls = self.vector_store.search_similar_ddls(summary, top_k=1)
            old_ddl_data = similar_ddls[0] if similar_ddls else None
        
        if not old_ddl_data:
            return "No matching DDL found in the database. Please store the original DDL first."
        
        old_ddl = old_ddl_data["ddl_content"]
        
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
        
        # Update stored DDL with new version
        self.vector_store.store_ddl(new_ddl, old_ddl_data["schema_name"])
        
        return response
    
    def generate_analytical_query(self, user_question, context_tables=None):
        """
        Generate analytical SQL query based on user question
        Uses similarity search to find relevant DDLs
        """
        # Search for relevant DDLs
        relevant_ddls = self.vector_store.search_similar_ddls(user_question, top_k=3)
        
        if not relevant_ddls:
            return "No relevant table schemas found in the database."
        
        # Build context with relevant DDLs
        ddl_context = "\n\n".join([
            f"Table: {ddl['table_name']}\n{ddl['ddl_content']}" 
            for ddl in relevant_ddls
        ])
        
        analytical_prompt = """You are an expert SQL analyst specializing in Snowflake.

Your task is to:
1. Understand the table structures from the provided DDLs
2. Interpret the user's analytical question
3. Generate an optimized Snowflake SQL query that answers their question

Guidelines:
- Use appropriate aggregations (SUM, AVG, COUNT, MIN, MAX)
- Include GROUP BY, HAVING, ORDER BY as needed
- Use window functions when appropriate (ROW_NUMBER, RANK, LAG, LEAD)
- Add CTEs (WITH clauses) for complex queries
- Include proper date/time functions for temporal analysis
- Use JOINs if multiple tables are involved
- Add comments explaining the query logic
- Optimize for Snowflake performance
- Format the query for readability

Return ONLY the SQL query with inline comments."""

        messages = [
            System(content=analytical_prompt),
            Human(content=f"""Available Table DDLs:
{ddl_context}

User Question: {user_question}

Generate an analytical SQL query that answers this question using the available tables.""")
        ]
        
        response = chat(messages)
        return response
    
    def generate_query_from_description(self, query_description, query_type="select"):
        """
        Generate any SQL query based on natural language description
        Uses similarity search to find relevant tables
        """
        # Search for relevant DDLs
        relevant_ddls = self.vector_store.search_similar_ddls(query_description, top_k=3)
        
        if not relevant_ddls:
            return "No relevant table schemas found in the database."
        
        # Build context
        ddl_context = "\n\n".join([
            f"Table: {ddl['table_name']}\n{ddl['ddl_content']}" 
            for ddl in relevant_ddls
        ])
        
        query_type_instructions = {
            "select": "Generate a SELECT query",
            "insert": "Generate an INSERT statement",
            "update": "Generate an UPDATE statement",
            "delete": "Generate a DELETE statement",
            "merge": "Generate a MERGE (upsert) statement"
        }
        
        messages = [
            System(content=self.system_prompt),
            Human(content=f"""Available Table DDLs:
{ddl_context}

Task: {query_type_instructions.get(query_type, 'Generate a SQL query')}

Description: {query_description}

Generate the appropriate SQL query based on the description and available tables.""")
        ]
        
        response = chat(messages)
        return response


# Example Usage
if __name__ == "__main__":
    # Initialize generator
    generator = DDLToSQLGenerator()
    
    # Step 1: Store initial DDLs
    print("=== Step 1: Storing Initial DDLs ===")
    
    customers_ddl = """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        email VARCHAR(100),
        created_at TIMESTAMP
    );
    """
    
    orders_ddl = """
    CREATE TABLE orders (
        order_id INT PRIMARY KEY,
        customer_id INT,
        order_date TIMESTAMP,
        total_amount DECIMAL(10,2),
        status VARCHAR(20)
    );
    """
    
    generator.store_ddl(customers_ddl, "public")
    generator.store_ddl(orders_ddl, "public")
    
    # Step 2: Update DDL and generate ALTER statements
    print("\n=== Step 2: Updating DDL and Generating ALTER Statements ===")
    
    updated_customers_ddl = """
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        email VARCHAR(100),
        phone VARCHAR(20),
        address VARCHAR(200),
        created_at TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    alter_statements = generator.generate_alter_statements(updated_customers_ddl, "customers")
    print(alter_statements)
    
    # Step 3: Generate analytical queries based on user questions
    print("\n=== Step 3: Generating Analytical Queries ===")
    
    question1 = "Show me the top 10 customers by total order value"
    query1 = generator.generate_analytical_query(question1)
    print(f"\nQuestion: {question1}")
    print(query1)
    
    question2 = "What is the average order amount by customer, showing only customers with more than 5 orders?"
    query2 = generator.generate_analytical_query(question2)
    print(f"\nQuestion: {question2}")
    print(query2)
    
    question3 = "Find customers who haven't placed any orders in the last 30 days"
    query3 = generator.generate_analytical_query(question3)
    print(f"\nQuestion: {question3}")
    print(query3)
    
    # Step 4: Generate specific query types
    print("\n=== Step 4: Generating Specific Query Types ===")
    
    insert_desc = "Insert a new customer with email and phone number"
    insert_query = generator.generate_query_from_description(insert_desc, "insert")
    print(f"\nDescription: {insert_desc}")
    print(insert_query)
    
    update_desc = "Update customer address for a specific customer ID"
    update_query = generator.generate_query_from_description(update_desc, "update")
    print(f"\nDescription: {update_desc}")
    print(update_query)
