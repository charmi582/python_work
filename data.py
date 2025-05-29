import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
from datetime import datetime

# MySQL連接設定
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="012345678",
    database="database_work"
)
cursor = conn.cursor()

root = tk.Tk()
root.title("水果倉庫系統")

# ==== 新增水果 ====
def add_fruit():
    name = fruit_name.get()
    category = fruit_category.get()
    desc = fruit_desc.get()
    unit = fruit_unit.get()
    if not name:
        messagebox.showerror("錯誤", "請輸入水果名稱")
        return
    cursor.execute("INSERT INTO market_fruit (name, category, description, unit) VALUES (%s, %s, %s, %s)",
                   (name, category, desc, unit))
    conn.commit()
    messagebox.showinfo("成功", f"已新增水果：{name}")

tk.Label(root, text="水果名稱").grid(row=0, column=0)
fruit_name = tk.Entry(root)
fruit_name.grid(row=0, column=1)
tk.Label(root, text="分類").grid(row=1, column=0)
fruit_category = tk.Entry(root)
fruit_category.grid(row=1, column=1)
tk.Label(root, text="描述").grid(row=2, column=0)
fruit_desc = tk.Entry(root)
fruit_desc.grid(row=2, column=1)
tk.Label(root, text="單位").grid(row=3, column=0)
fruit_unit = tk.Entry(root)
fruit_unit.grid(row=3, column=1)
tk.Button(root, text="新增水果", command=add_fruit).grid(row=4, column=1)

# ==== 新增進貨 ====
def add_inbound():
    fruit_id = inbound_fruit_id.get()
    quantity = inbound_qty.get()
    supplier = inbound_supplier.get()
    unit_cost = inbound_cost.get()
    if not fruit_id or not quantity:
        messagebox.showerror("錯誤", "請填入完整資訊")
        return
    cursor.execute("INSERT INTO market_inbound (fruit_id, quantity, supplier, unit_cost, inbound_date) VALUES (%s, %s, %s, %s, %s)",
                   (fruit_id, quantity, supplier, unit_cost, datetime.now()))
    cursor.execute("SELECT id FROM market_stock WHERE fruit_id=%s", (fruit_id,))
    stock = cursor.fetchone()
    if stock:
        cursor.execute("UPDATE market_stock SET quantity=quantity+%s, last_updated=%s WHERE fruit_id=%s",
                       (quantity, datetime.now(), fruit_id))
    else:
        cursor.execute("INSERT INTO market_stock (fruit_id, quantity, storage_location, last_updated) VALUES (%s, %s, %s, %s)",
                       (fruit_id, quantity, "倉庫A", datetime.now()))
    conn.commit()
    messagebox.showinfo("成功", f"已進貨 {quantity}")

tk.Label(root, text="進貨水果ID").grid(row=5, column=0)
inbound_fruit_id = tk.Entry(root)
inbound_fruit_id.grid(row=5, column=1)
tk.Label(root, text="數量").grid(row=6, column=0)
inbound_qty = tk.Entry(root)
inbound_qty.grid(row=6, column=1)
tk.Label(root, text="供應商").grid(row=7, column=0)
inbound_supplier = tk.Entry(root)
inbound_supplier.grid(row=7, column=1)
tk.Label(root, text="單價").grid(row=8, column=0)
inbound_cost = tk.Entry(root)
inbound_cost.grid(row=8, column=1)
tk.Button(root, text="新增進貨", command=add_inbound).grid(row=9, column=1)

# ==== 新增出貨 ====
def add_outbound():
    fruit_id = outbound_fruit_id.get()
    quantity = outbound_qty.get()
    customer = outbound_customer.get()
    unit_price = outbound_price.get()
    if not fruit_id or not quantity:
        messagebox.showerror("錯誤", "請填入完整資訊")
        return
    cursor.execute("SELECT quantity FROM market_stock WHERE fruit_id=%s", (fruit_id,))
    stock = cursor.fetchone()
    if stock and stock[0] >= int(quantity):
        cursor.execute("INSERT INTO market_outbound (fruit_id, quantity, customer, unit_price, outbound_date) VALUES (%s, %s, %s, %s, %s)",
                       (fruit_id, quantity, customer, unit_price, datetime.now()))
        cursor.execute("UPDATE market_stock SET quantity=quantity-%s, last_updated=%s WHERE fruit_id=%s",
                       (quantity, datetime.now(), fruit_id))
        conn.commit()
        messagebox.showinfo("成功", f"已出貨 {quantity}")
    else:
        messagebox.showerror("錯誤", f"庫存不足（目前庫存：{stock[0] if stock else 0}）")

tk.Label(root, text="出貨水果ID").grid(row=10, column=0)
outbound_fruit_id = tk.Entry(root)
outbound_fruit_id.grid(row=10, column=1)
tk.Label(root, text="數量").grid(row=11, column=0)
outbound_qty = tk.Entry(root)
outbound_qty.grid(row=11, column=1)
tk.Label(root, text="客戶").grid(row=12, column=0)
outbound_customer = tk.Entry(root)
outbound_customer.grid(row=12, column=1)
tk.Label(root, text="單價").grid(row=13, column=0)
outbound_price = tk.Entry(root)
outbound_price.grid(row=13, column=1)
tk.Button(root, text="新增出貨", command=add_outbound).grid(row=14, column=1)

# ==== 顯示庫存 ====
def show_stock():
    stock_win = tk.Toplevel(root)
    stock_win.title("庫存清單")
    tree = ttk.Treeview(stock_win, columns=("水果", "數量", "儲存位置", "更新時間"), show="headings")
    for col in ("水果", "數量", "儲存位置", "更新時間"):
        tree.heading(col, text=col)
    tree.pack()
    cursor.execute("""
        SELECT f.name, s.quantity, s.storage_location, s.last_updated
        FROM market_stock s JOIN market_fruit f ON s.fruit_id = f.id
    """)
    for row in cursor.fetchall():
        tree.insert("", tk.END, values=row)

tk.Button(root, text="查看庫存", command=show_stock).grid(row=15, column=1)

root.mainloop()
cursor.close()
conn.close()
