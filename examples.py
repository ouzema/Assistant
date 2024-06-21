from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

examples = [
    {
        "input": "What is the total quantity of product X?",
        "query": "SELECT SUM(quantity) FROM products WHERE name = 'X';"
    },
    {
        "input": "How many orders were placed by customer Y?",
        "query": "SELECT COUNT(*) FROM orders WHERE customer_id = (SELECT id FROM customers WHERE name = 'Y');"
    },
    {
        "input": "Who are the customers who have made a purchase in the last week.",
        "query": "SELECT DISTINCT customers.name FROM customers JOIN orders ON customers.id = orders.customer_id WHERE orders.created_at >= CURRENT_DATE - INTERVAL '1 week';"
    },
    {
        "input": "Show me the average price of products in category 'Electronics'.",
        "query": "SELECT AVG(price) FROM products WHERE category = 'Electronics';"
    },
    {
        "input": "Find the most popular payment method used by customers.",
        "query": "SELECT payment_method FROM (SELECT payment_method, COUNT(*) AS method_count FROM orders WHERE payment_method IS NOT NULL GROUP BY payment_method) AS popular_payment_method ORDER BY COUNT(*) DESC LIMIT 1;"
    },
    {
        "input": "Show me the order details of 'X'.",
        "query": "SELECT products.name, workorders.created_at FROM workorders JOIN orders ON workorders.order_id = orders.id JOIN customers ON orders.customer_id = customers.id JOIN products ON workorders.product_id = products.id WHERE customers.name = 'X';"
    },
    {
        "input": "Which orders contain products with a quantity less than 10?",
        "query": "SELECT DISTINCT orders.id FROM orders JOIN order_items ON orders.id = order_items.order_id WHERE order_items.quantity < 10;"
    },
    {
        "input": "How many customers have made a purchase in the last month?",
        "query": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE_PART('month', CURRENT_DATE - INTERVAL '1 month') = DATE_PART('month', created_at);"
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;"
    },
    {
        "input": "Show me the total revenue for each product category.",
        "query": "SELECT category, SUM(total_amount) AS revenue FROM orders JOIN order_items ON orders.id = order_items.order_id JOIN products ON order_items.product_id = products.id GROUP BY category;"
    },
    {
        "input": "What is in production?",
        "query": "SELECT products.name AS product_name, workorders.created_at AS created_date, customers.name AS customer_name, workorders.quantity FROM workorders JOIN products ON workorders.product_id = products.id JOIN orders ON workorders.order_id = orders.id JOIN customers ON orders.customer_id = customers.id WHERE workorders.status = '1';"
    },
    {
        "input": "What's your name?",
        "query": "I am Leanios_core Assistant"
    },
    {
        "input": "Who are you?",
        "query": "I am your Assistant"
    },
    {
        "input": "What is the stock level per warehouse?",
        "query": "SELECT warehouses.name, warehouses_products.quantity FROM warehouses JOIN warehouses_products ON warehouses.id = warehouses_products.warehouse_id ORDER BY warehouses.name;"
    },
    {
        "input": "What can you do?",
        "query": "I can answer your questions or inquiries about the Workflow, the Production or any other related process"
    },
    {
        "input": "Hi",
        "query": "Hello there, how may I assist you today?"
    },
    {
        "input": "Hello",
        "query": "Greetings! How may I be at your service you today?"
    },
    {
        "input": "What are the OF that are pending",
        "query": "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 0;"
    },
    {
        "input": "What are the OF that are stopping",
        "query": "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 2;"
    },
    {
        "input": "What are the OF that are completed",
        "query": "SELECT products.name FROM workorders JOIN products ON workorders.product_id = products.id WHERE workorders.status = 4;"
    },
    {
        "input": "What is the number of client orders for this month?",
        "query": "SELECT customer_id, COUNT(*) AS number_of_orders FROM orders WHERE EXTRACT(MONTH FROM created_at) = EXTRACT(MONTH FROM CURRENT_DATE) GROUP BY customer_id;"
    },
    {
        "input": "Who works on the machine 'X'",
        "query": "SELECT operators.full_name FROM operators JOIN machines_operators ON operators.id = machines_operators.operator_id JOIN machines ON machines.id = machines_operators.machine_id WHERE machines.name = 'X';"
    },
    {
        "input": "what is the production rate by item?",
        "query": "SELECT products.name AS 'Product Name', COUNT(productions.id) AS 'Number of Productions', SUM(productions.quantity) AS 'Total Quantity Produced', AVG(productions.performance) AS 'Average Production Rate' FROM productions JOIN products ON productions.product_id = products.id GROUP BY products.name;"
    },
    {
        "input": "what is the rate of non-compliance",
        "query": "SELECT downtime_issues.name, COUNT(downtimes.id), SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END), (SUM(CASE WHEN downtimes.status = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(downtimes.id)) FROM downtimes JOIN downtime_issues ON downtimes.downtime_issue_id = downtime_issues.id GROUP BY downtime_issues.name;"
    },
    {
        "input": "what raw material stocks will expire this month?",
        "query": "SELECT products.name, warehouses.name, inventory_items.quantity, inventory_items.created_at, inventory_items.created_at + INTERVAL '1 month' FROM inventory_items JOIN products ON inventory_items.product_id = products.id JOIN warehouses ON inventory_items.warehouse_id = warehouses.id WHERE products.is_final_product = FALSE AND products.expiration_date IS NOT NULL AND EXTRACT(MONTH FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(MONTH FROM CURRENT_DATE) AND EXTRACT(YEAR FROM (inventory_items.created_at + INTERVAL '1 month')) = EXTRACT(YEAR FROM CURRENT_DATE);"
    },
    {
        "input": "What are the products categories?",
        "query": "SELECT products.name AS product_name, product_categories.name AS category_name FROM products JOIN product_categories ON products.product_category_id = product_categories.id;"
    },
    {
        "input": "What are the machine areas?",
        "query": "SELECT machines.name AS machine_name, areas.name AS area_name FROM machines JOIN areas ON machines.area_id = areas.id;"
    },
    {
        "input": "what are the Internal OF?",
        "query": "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = true;"
    },
    {
        "input": "what are the external OF?",
        "query": "SELECT products.name FROM products JOIN workorders ON products.id = workorders.product_id WHERE workorders.internal = false;"
    },
    {
        "input": "what raw material stocks will expire this month?",
        "query": "SELECT products.name, batches.expiration_date FROM products JOIN batches ON products.id = batches.product_id JOIN product_categories ON products.product_category_id = product_categories.id WHERE product_categories.raw_material = true AND DATE_TRUNC('month', batches.expiration_date) = DATE_TRUNC('month', CURRENT_DATE) ORDER BY batches.expiration_date;"
    },
    {
        "input": "How many semi finished products do I have in stock?",
        "query": "SELECT name, quantity FROM dummy.products WHERE is_semi_finished = true;"
    },
    {
        "input": "Show me the products that have been delivered",
        "query": "SELECT products.name, customers.name, delivery_notes.created_at FROM delivery_notes JOIN workorders ON workorders.order_id = delivery_notes.order_id JOIN products ON products.id = workorders.product_id JOIN orders ON orders.id = workorders.order_id JOIN customers ON customers.id = orders.customer_id;"
    },
    {
        "input": "who are the admin users?",
        "query": "SELECT users.* FROM users JOIN users_roles ON users.id = users_roles.user_id JOIN roles ON users_roles.role_id = roles.id WHERE roles.name = 'administrator';"
    },
    {
        "input": "who are the production admins?",
        "query": "SELECT users.* FROM users JOIN users_roles ON users.id = users_roles.user_id JOIN roles ON users_roles.role_id = roles.id WHERE roles.name = 'production_admin';"
    },
    {
        "input": "who are the orders admins?",
        "query": "SELECT users.* FROM users JOIN users_roles ON users.id = users_roles.user_id JOIN roles ON users_roles.role_id = roles.id WHERE roles.name = 'orders_admin';"
    },
    {
        "input": "who are the stock admins?",
        "query": "SELECT users.* FROM users JOIN users_roles ON users.id = users_roles.user_id JOIN roles ON users_roles.role_id = roles.id WHERE roles.name = 'stock_admin';"
    },
    {
        "input": "who are the finance users?",
        "query": "SELECT users.* FROM users JOIN users_roles ON users.id = users_roles.user_id JOIN roles ON users_roles.role_id = roles.id WHERE roles.name = 'finances';"
    },
    {
        "input": "Show me all available waste types.",
        "query": "SELECT DISTINCT waste_type FROM wastes;"
    },
    {
        "input": "List all the active subscriptions.",
        "query": "SELECT * FROM subscriptions WHERE state = 1;"
    },
    {
        "input": "Retrieve the names of all the suppliers.",
        "query": "SELECT name FROM suppliers;"
    },
    {
        "input": "What are the names of the active teams?",
        "query": "SELECT name FROM teams WHERE archived = false;"
    },
    {
        "input": "Display the details of all the products.",
        "query": "SELECT * FROM products;"
    },
    {
        "input": "Retrieve the names of all the warehouses.",
        "query": "SELECT name FROM warehouses;"
    },
    {
        "input": "List all the products categorized as final products.",
        "query": "SELECT * FROM products WHERE is_final_product = true;"
    },
    {
        "input": "Retrieve the names of all the areas.",
        "query": "SELECT name FROM areas;"
    },
    {
        "input": "Display the names of all the portal products.",
        "query": "SELECT portal_products.* FROM portal_products JOIN portals ON portal_products.portal_id = portals.id;"
    },
    {
        "input": "List all customers who have not made any purchase.",
        "query": "SELECT name FROM customers WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders);"
    },
    {
        "input": "What is the highest selling product in the last year?",
        "query": "SELECT product_id, SUM(quantity) as total_sold FROM order_items WHERE created_at >= DATE_TRUNC('year', CURRENT_DATE) GROUP BY product_id ORDER BY total_sold DESC LIMIT 1;"
    },
    {
        "input": "Which product category has the most products?",
        "query": "SELECT category, COUNT(*) as product_count FROM products GROUP BY category ORDER BY product_count DESC LIMIT 1;"
    },
    {
        "input": "What is the total revenue generated by each product category?",
        "query": "SELECT category, SUM(price * quantity) as total_revenue FROM products JOIN order_items ON products.id = order_items.product_id GROUP BY category;"
    },
    {
        "input": "List the top 5 customers who have spent the most in the last year.",
        "query": "SELECT customers.name, SUM(order_items.price * order_items.quantity) as total_spent FROM customers JOIN orders ON customers.id = orders.customer_id JOIN order_items ON orders.id = order_items.order_id WHERE orders.created_at >= DATE_TRUNC('year', CURRENT_DATE) GROUP BY customers.name ORDER BY total_spent DESC LIMIT 5;"
    },
    {
        "input": "Find all products that have never been ordered.",
        "query": "SELECT name FROM products WHERE id NOT IN (SELECT DISTINCT product_id FROM order_items);"
    },
    {
        "input": "What is the average order value in the last month?",
        "query": "SELECT AVG(order_total) FROM (SELECT SUM(order_items.price * order_items.quantity) as order_total FROM orders JOIN order_items ON orders.id = order_items.order_id WHERE orders.created_at >= CURRENT_DATE - INTERVAL '1 month' GROUP BY orders.id) as monthly_orders;"
    },
    {
        "input": "List the top 3 most frequently ordered products.",
        "query": "SELECT products.name, COUNT(workorders.product_id) AS order_count FROM workorders JOIN products ON workorders.product_id = products.id GROUP BY products.name ORDER BY order_count DESC LIMIT 3;"
    },
    {
        "input": "Find the total number of orders placed in each month of the current year.",
        "query": "SELECT DATE_TRUNC('month', created_at) as month, COUNT(*) as total_orders FROM orders WHERE created_at >= DATE_TRUNC('year', CURRENT_DATE) GROUP BY month ORDER BY month;"
    },
    {
        "input": "Which customers have placed more than 5 orders?",
        "query": "SELECT customers.name, COUNT(orders.id) as order_count FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY customers.name HAVING COUNT(orders.id) > 5;"
    },
    {
        "input": "What is the most expensive product in each category?",
        "query": "SELECT products.product_category_id, products.name AS product_name, prices.amount AS price FROM products JOIN prices ON products.id = prices.product_id WHERE prices.amount = (SELECT MAX(prices.amount) FROM prices WHERE prices.product_id = products.id AND prices.archived = false) ORDER BY products.product_category_id;"
    },
    {
        "input": "Which customers have placed at least 1 order?",
        "query": "SELECT customers.name, COUNT(orders.id) as order_count FROM customers JOIN orders ON customers.id = orders.customer_id GROUP BY customers.name HAVING COUNT(orders.id) >= 1;"
    }
]
