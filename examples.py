from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

examples = [
    {"input": "What is the total quantity of product X?",
      "query": "SELECT SUM(quantity) FROM products WHERE name = 'X';"},
    {
        "input": "How many orders were placed by customer Y?",
        "query": "SELECT COUNT(*) FROM orders WHERE customer_id = (SELECT id FROM customers WHERE name = 'Y');",
    },
    {
        "input": "Who are the customers who have made a purchase in the last week.",
        "query": "SELECT DISTINCT customers.name FROM customers JOIN orders ON customers.id = orders.customer_id WHERE orders.created_at >= CURRENT_DATE - INTERVAL '1 week';",
    },
    {
        "input": "Show me the average price of products in category 'Electronics'.",
        "query": "SELECT AVG(price) FROM products WHERE category = 'Electronics';",
    },
    {
        "input": "Find the most popular payment method used by customers.",
        "query": "SELECT payment_method FROM ( SELECT payment_method, COUNT(*) AS method_count FROM orders WHERE payment_method IS NOT NULL GROUP BY payment_method) AS popular_payment_method ORDER BY COUNT(*) DESC LIMIT 1;",
    },
    {
        "input": "Show me the order details 'X'.",
        "query": "SELECT products.name, workorders.created_at FROM workorders JOIN orders ON workorders.order_id = orders.id JOIN customers ON orders.customer_id = customers.id JOIN products ON workorders.product_id = products.id WHERE customers.name = 'X';",
    },
    {
        "input": "Which orders contain products with a quantity less than 10?",
        "query": "SELECT DISTINCT orders.id FROM orders JOIN order_items ON orders.id = order_items.order_id WHERE order_items.quantity < 10;",
    },
    {
        "input": "How many customers have made a purchase in the last month?",
        "query": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE_PART('month', CURRENT_DATE - INTERVAL '1 month') = DATE_PART('month', created_at);",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Show me the total revenue for each product category.",
        "query": "SELECT category, SUM(total_amount) AS revenue FROM orders JOIN order_items ON orders.id = order_items.order_id JOIN products ON order_items.product_id = products.id GROUP BY category;",
    },
    {
        "input" : "What's in productions?",
        "query" : "SELECT products.name, customers.name, workorders.created_at FROM workorders INNER JOIN products ON workorders.product_id = products.id INNER JOIN orders ON workorders.order_id = orders.id INNER JOIN customers ON orders.customer_id = customers.id WHERE workorders.status = 1;"
    },
    
    {
        "input" : "What's your name?",
        "query" : "I am Leanios_Core Assistant"
    },
    {
        "input" : "Who are you?",
        "query" : "I am your Assistant"
    },
    {
        "input" : "What is the stock level per warehouse?",
        "query" : "SELECT warehouses.name, warehouses_products.quantity FROM warehouses JOIN warehouses_products ON warehouses.id = warehouses_products.warehouse_id ORDER BY warehouses.name;"
    },
    {
        "input" : "What can you do?",
        "query" : "I can answer your questions or inquiries about the Workflow, the Production or any other related process"
    },
    {
        "input" : "Hi",
        "query" : "Hello there, how may I assist you today?"
    },
    {
        "input" : "Hello",
        "query" : "Greetings! How may I be at your service you today?"
    },
    {
        "input" : "What are the OF that are pending",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 0;"
    },
        {
        "input" : "What are the OF that are stopping",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 2;"
    },
    {
        "input" : "What are the OF that are completed",
        "query" : "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 4;"
    },
    {
        "input" : "What is the number of client orders for this month?",
        "query" : "SELECT customer_id, COUNT(*) AS number_of_orders FROM orders WHERE EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE) GROUP BY customer_id;"
    },
    {
        "input" : "Who works on the machine 'X'",
        "query" : "SELECT operators.full_name FROM operators JOIN machines_operators ON operators.id = machines_operators.operator_id JOIN machines ON machines.id = machines_operators.machine_id WHERE machines.name = 'X';"
    },
    {
        "input" : "what is the production rate by item?",
        "query" : "SELECT products.name AS 'Product Name', COUNT(productions.id) AS 'Number of Productions',SUM(productions.quantity) AS 'Total Quantity Produced', AVG(productions.performance) AS 'Average Production Rate' FROM productions JOIN products ON productions.product_id = products.id GROUP BY products.name;"
    },
    {
        "input" : "what is the rate of non-compliance",
        "query" : "SELECT downtime_issues.name, COUNT(downtimes.id), SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END), (SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(downtimes.id)) FROM downtimes JOIN downtime_issues ON downtimes.downtime_issue_id = downtime_issues.id GROUP BY downtime_issues.name;"
    },
    {
        "input" : "what raw material stocks will expire this month?",
        "query" : "SELECT products.name, warehouses.name, inventory_items.quantity, inventory_items.created_at, inventory_items.created_at + INTERVAL '1 month' FROM inventory_items JOIN products ON inventory_items.product_id = products.id JOIN warehouses ON inventory_items.warehouse_id = warehouses.id WHERE products.is_final_product = FALSE AND products.expiration_date IS NOT NULL AND EXTRACT(MONTH FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(YEAR FROM CURRENT_DATE);"
    },
    {
        "input" : "What are the products categories?",
        "query" : "SELECT products.name AS product_name, product_categories.name AS category_name FROM products JOIN product_categories ON products.product_category_id = product_categories.id;"
    },
    {
        "input" : "What are the machine areas?",
        "query" : "SELECT machines.name AS machine_name, areas.name AS area_name FROM machines JOIN areas ON machines.area_id = areas.id;"
    },
    {
        "input" : "what are the Internal OF?",
        "query" : "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = true;"
    },
    {
        "input" : "what are the external OF?",
        "query" : "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = false;"
    },
    {
        "input" : "what raw material stocks will expire this month?",
        "query" : "SELECT products.name, batches.expiration_date FROM products JOIN batches ON products.id = batches.product_id JOIN product_categories ON products.product_category_id = product_categories.id WHERE product_categories.raw_material = true AND DATE_TRUNC('month', batches.expiration_date) = DATE_TRUNC('month', CURRENT_DATE) ORDER BY batches.expiration_date;"
    },
    {
        "input" : "How many semi finished products do I have in stock?",
        "query" : "SELECT name, quantity FROM dummy.products WHERE is_semi_finished = true;"
    },
    {
        "input" : "Who are the amdin users?",
        "query" : "SELECT firstname, lastname FROM users WHERE admin = TRUE;"
    },
    {
        "input" : "Who are the amdin users?",
        "query" : "SELECT firstname, lastname FROM users WHERE admin = TRUE;"
    },
    {
        "input" : "Show me the products that have been delivered",
        "query" : "SELECT products.name, customers.name, delivery_notes.created_at FROM delivery_notes JOIN workorders ON workorders.order_id = delivery_notes.order_id JOIN products ON products.id = workorders.product_id JOIN orders ON orders.id = workorders.order_id JOIN customers ON customers.id = orders.customer_id;"
    }
]