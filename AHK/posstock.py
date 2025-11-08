# posstock.py

import threading
import time
from datetime import datetime

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import NumericProperty, StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen, ScreenManager

from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDIconButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.recycleview.views import RecycleDataViewBehavior

# KV Language Definitions
KV = '''
<CompanyItem>:
    orientation: 'horizontal'
    size_hint_y: None
    height: dp(80)
    padding: dp(10)
    spacing: dp(10)
    # Removed padding
    elevation: 3

    MDCard:
        orientation: 'horizontal'
        elevation: 4
        size_hint: 1, 1
        padding: dp(10)
        spacing: dp(10)
        ripple_behavior: True
        on_press: app.root.get_screen('company_list_screen').select_company(root.name)

        MDLabel:
            text: root.name
            theme_text_color: "Primary"
            size_hint_x: 0.6
            halign: "left"
            valign: "middle"
            text_size: self.size

        MDIconButton:
            icon: "information"
            theme_text_color: "Primary"
            size_hint_x: 0.2
            on_release: app.root.get_screen('company_list_screen').show_company_info(root.name)

        MDIconButton:
            icon: "delete"
            theme_text_color: "Error"
            size_hint_x: 0.2
            on_release: app.root.get_screen('company_list_screen').delete_company(root.name)


<ProductItem>:
    orientation: 'horizontal'
    size_hint_y: None
    height: dp(80)
    padding: dp(10)
    spacing: dp(10)
    # Removed padding
    elevation: 2

    MDCard:
        orientation: 'horizontal'
        elevation: 5
        size_hint: 1, 1
        padding: dp(10)
        spacing: dp(10)
        ripple_behavior: True
        on_press: app.root.get_screen('product_list_screen').select_product(root.name)

        MDLabel:
            text: root.name
            theme_text_color: "Primary"
            size_hint_x: 0.5
            halign: "left"
            valign: "middle"
            text_size: self.size

        MDLabel:
            text: "Qty: {}".format(root.quantity)
            theme_text_color: "Secondary"
            size_hint_x: 0.2
            halign: "center"
            valign: "middle"
            text_size: self.size

        MDLabel:
            text: root.price_tax_text
            theme_text_color: "Secondary"
            size_hint_x: 0.2
            halign: "right"
            valign: "middle"
            text_size: self.size

        MDIconButton:
            icon: "information"
            theme_text_color: "Primary"
            size_hint_x: 0.05
            on_release: app.root.get_screen('product_list_screen').show_product_info(root.name)

        MDIconButton:
            icon: "delete"
            theme_text_color: "Error"
            size_hint_x: 0.05
            on_release: app.root.get_screen('product_list_screen').delete_product(root.name)


<TransactionItem>:
    orientation: 'vertical'
    size_hint_y: None
    height: dp(60)
    padding: dp(10)
    elevation: 2
    ripple_behavior: False

    MDLabel:
        text: root.text
        theme_text_color: "Primary"
        font_style: "Subtitle1"
        size_hint_y: None
        height: self.texture_size[1]
        halign: "left"
        valign: "middle"
        text_size: self.width, None

    MDLabel:
        text: root.secondary_text
        theme_text_color: "Secondary"
        font_style: "Body2"
        size_hint_y: None
        height: self.texture_size[1]
        halign: "left"
        valign: "middle"
        text_size: self.width, None


<NoProductItem>:
    orientation: 'horizontal'
    size_hint_y: None
    height: dp(80)
    padding: dp(10)

    MDLabel:
        text: root.message
        theme_text_color: "Secondary"
        halign: "center"
        valign: "middle"
        text_size: self.size


<CompanyListView>:
    viewclass: 'CompanyItem'
    RecycleBoxLayout:
        default_size: None, dp(80)
        default_size_hint: 1, None
        size_hint_y: None
        orientation: 'vertical'
        spacing: dp(2)
        padding: dp(10)
        height: self.minimum_height


<ProductListView>:
    viewclass: 'ProductItem'
    RecycleBoxLayout:
        default_size: None, dp(80)
        default_size_hint: 1, None
        size_hint_y: None
        orientation: 'vertical'
        spacing: dp(2)
        padding: dp(10)
        height: self.minimum_height


<TransactionListView>:
    viewclass: 'TransactionItem'
    RecycleBoxLayout:
        default_size: None, dp(60)
        default_size_hint: 1, None
        size_hint_y: None
        orientation: 'vertical'
        spacing: dp(2)
        padding: dp(10)
        height: self.minimum_height
'''

# Load KV string
Builder.load_string(KV)

# Custom RecycleView Classes
class CompanyListView(RecycleView):
    """RecycleView for displaying companies."""
    pass  # No need to redefine 'data'


class ProductListView(RecycleView):
    """RecycleView for displaying products."""
    pass  # No need to redefine 'data'


class TransactionListView(RecycleView):
    """RecycleView for displaying transactions."""
    pass  # No need to redefine 'data'


# Custom RecycleView Item Classes
class CompanyItem(RecycleDataViewBehavior, ButtonBehavior, BoxLayout):
    """Custom RecycleView item for a company."""
    name = StringProperty()
    details = StringProperty()


class ProductItem(RecycleDataViewBehavior, ButtonBehavior, BoxLayout):
    """Custom RecycleView item for a product."""
    name = StringProperty()
    quantity = NumericProperty()
    price = NumericProperty()
    tax = NumericProperty()
    price_tax_text = StringProperty()
    is_dummy = BooleanProperty(False)  # Property to indicate dummy items

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_price_tax()

    def on_price(self, instance, value):
        self.update_price_tax()

    def on_tax(self, instance, value):
        self.update_price_tax()

    def update_price_tax(self):
        """Updates the price and tax text."""
        self.price_tax_text = "Price: ${0:.2f}\nTax: {1:.1f}%".format(self.price, self.tax)

    def on_press(self):
        if not self.is_dummy:
            super().on_press()


class NoProductItem(RecycleDataViewBehavior, BoxLayout):
    """Custom RecycleView item for displaying no products found."""
    message = StringProperty()


class TransactionItem(RecycleDataViewBehavior, ButtonBehavior, BoxLayout):
    """Custom RecycleView item for a transaction."""
    text = StringProperty()
    secondary_text = StringProperty()


# Dialog Content Classes
class AddCompanyContent(BoxLayout):
    """Layout for the Add Company dialog."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.size_hint_y = None
        self.height = dp(120)

        self.comp_name = MDTextField(
            hint_text="Company Name",
            required=True,
            helper_text="Enter company name",
            helper_text_mode="on_focus",
            size_hint_x=1
        )
        self.comp_details = MDTextField(
            hint_text="Company Details",
            required=False,
            helper_text="Enter company details",
            helper_text_mode="on_focus",
            size_hint_x=1
        )
        self.add_widget(self.comp_name)
        self.add_widget(self.comp_details)


class AddProductContent(BoxLayout):
    """Layout for the Add Product dialog."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.size_hint_y = None
        self.height = dp(240)  # Increased height

        self.product_name = MDTextField(
            hint_text="Product Name",
            required=True,
            helper_text="Enter product name",
            helper_text_mode="on_focus",
            size_hint_x=1
        )
        self.product_qty = MDTextField(
            hint_text="Initial Quantity",
            required=True,
            helper_text="Enter initial quantity",
            helper_text_mode="on_focus",
            input_filter='int',
            size_hint_x=1
        )
        self.product_price = MDTextField(
            hint_text="Price (e.g., 19.99)",
            required=True,
            helper_text="Enter product price",
            helper_text_mode="on_focus",
            input_filter='float',
            size_hint_x=1
        )
        self.product_tax = MDTextField(
            hint_text="Tax Percentage (e.g., 5.0)",
            required=False,
            helper_text="Enter tax percentage (GST/VAT)",
            helper_text_mode="on_focus",
            input_filter='float',
            size_hint_x=1
        )
        self.add_widget(self.product_name)
        self.add_widget(self.product_qty)
        self.add_widget(self.product_price)
        self.add_widget(self.product_tax)


# Screen Classes
class CompanyListScreen(Screen):
    """Screen to display and manage companies."""

    def on_pre_enter(self, *args):
        super().on_pre_enter(*args)
        self.build_ui()
        self.load_companies()

    def build_ui(self):
        """Builds the UI for the CompanyListScreen."""
        layout = BoxLayout(orientation='vertical')

        # Toolbar with enhanced color
        toolbar = MDTopAppBar(
            title="Companies",
            elevation=10,
            pos_hint={"top": 1},
            md_bg_color=MDApp.get_running_app().theme_cls.primary_color,
            left_action_items=[["arrow-left", lambda x: self.go_back()]],
            right_action_items=[["plus", lambda x: self.open_add_company_dialog()]]
        )
        layout.add_widget(toolbar)

        # RecycleView for company list
        self.company_list = CompanyListView()
        layout.add_widget(self.company_list)

        # Spinner Overlay (optional for loading companies)
        self.spinner = MDSpinner(
            size_hint=(None, None),
            size=(dp(46), dp(46)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            active=False
        )
        layout.add_widget(self.spinner)

        self.add_widget(layout)

    def load_companies(self):
        """Loads companies from Firebase and displays them."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        print(f"[DEBUG] Loading companies for user_id: {user_id}")  # Debugging

        if not user_id:
            self.show_error_dialog("Authentication Error", "User not authenticated.")
            print("[DEBUG] User not authenticated.")  # Debugging
            return

        # Activate spinner
        self.spinner.active = True

        # Start a new thread to fetch companies
        threading.Thread(target=self.fetch_companies, args=(user_id,), daemon=True).start()

    def fetch_companies(self, user_id):
        """Fetches companies from Firebase in a separate thread."""
        app = MDApp.get_running_app()
        try:
            companies = app.db.child("users").child(user_id).child("companies").get().val()
            print(f"[DEBUG] Companies fetched from Firebase: {companies}")  # Debugging
            companies_data = []

            if companies:
                for key, details in companies.items():
                    name = details.get("name", "Unnamed Company")
                    comp_details = details.get("details", "")
                    companies_data.append({"name": name, "details": comp_details})
            else:
                companies_data.append({"name": "No companies found. Add a new company.", "details": ""})

            # Schedule adding widgets to the UI thread
            Clock.schedule_once(lambda dt: self.update_company_list(companies_data), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Loading Companies", str(e)), 0)
            print(f"[DEBUG] Exception in fetch_companies: {e}")  # Debugging
        finally:
            # Deactivate spinner on the UI thread
            Clock.schedule_once(lambda dt: setattr(self.spinner, 'active', False), 0)

    def update_company_list(self, companies_data):
        """Updates the company list in the UI thread."""
        print("[DEBUG] Updating company list with RecycleView data.")
        self.company_list.data = [
            {"name": company["name"], "details": company["details"]} for company in companies_data
        ]
        self.company_list.refresh_from_data()  # Force refresh
        print("[DEBUG] Company list updated.")

    def open_add_company_dialog(self):
        """Opens a dialog to add a new company."""
        self.add_company_dialog = MDDialog(
            title="Add Company",
            type="custom",
            content_cls=AddCompanyContent(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: self.add_company_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="SAVE",
                    on_release=lambda x: self.save_company(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                ),
            ],
        )
        self.add_company_dialog.open()

    def save_company(self):
        """Saves the new company to Firebase."""
        name = self.add_company_dialog.content_cls.comp_name.text.strip()
        details = self.add_company_dialog.content_cls.comp_details.text.strip()

        # Enhanced validation
        if not name:
            self.show_error_dialog("Validation Error", "Company name cannot be empty.")
            print("[DEBUG] Validation Error: Company name is empty.")
            return

        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        print(f"[DEBUG] Saving company '{name}' for user_id: {user_id}")  # Debugging

        try:
            # Using company name as key; ensure uniqueness or consider using push()
            app.db.child("users").child(user_id).child("companies").child(name).set({
                "name": name,
                "details": details,
                "created_at": int(time.time())
            })
            print(f"[DEBUG] Company '{name}' saved successfully.")  # Debugging
            self.add_company_dialog.dismiss()
            self.load_companies()
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Saving Company", str(e)), 0)
            print(f"[DEBUG] Exception in save_company: {e}")  # Debugging

    def select_company(self, company_name):
        """Selects a company and navigates to the product list screen."""
        app = MDApp.get_running_app()
        app.sm.active_company = company_name  # Ensure this is a string
        app.sm.active_product = ""  # Reset active product to prevent residual data
        print(f"[DEBUG] Active company set to: {company_name}")  # Debugging
        app.sm.current = "product_list_screen"

    def delete_company(self, company_name):
        """Deletes a company from Firebase and refreshes the list."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()

        # Confirmation Dialog
        confirm_dialog = MDDialog(
            title="Delete Company",
            text=f"Are you sure you want to delete '{company_name}'?",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: confirm_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="DELETE",
                    on_release=lambda x: self.confirm_delete_company(confirm_dialog, company_name),
                    md_bg_color=MDApp.get_running_app().theme_cls.error_color
                ),
            ],
        )
        confirm_dialog.open()

    def confirm_delete_company(self, dialog, company_name):
        """Confirms and deletes the company."""
        dialog.dismiss()
        try:
            app = MDApp.get_running_app()
            user_id = app.user_id  # Correctly retrieve user_id
            app.db.child("users").child(user_id).child("companies").child(company_name).remove()
            print(f"[DEBUG] Company '{company_name}' deleted successfully.")
            self.load_companies()
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Deleting Company", str(e)), 0)
            print(f"[DEBUG] Exception in delete_company: {e}")  # Debugging

    def show_company_info(self, company_name):
        """Displays detailed information about the selected company."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()

        try:
            # Fetch company details from Firebase
            company_data = app.db.child("users").child(user_id).child("companies") \
                                  .child(company_name).get().val()

            if not company_data:
                self.show_error_dialog("Error", "Company data not found.")
                print("[DEBUG] Company data not found.")  # Debugging
                return

            name = company_data.get("name", "Unnamed Company")
            details = company_data.get("details", "No details provided.")
            created_at_ts = company_data.get("created_at", 0)
            created_at = datetime.fromtimestamp(created_at_ts).strftime('%Y-%m-%d %H:%M:%S')

            info_text = (
                f"Name: {name}\n"
                f"Details: {details}\n"
                f"Created At: {created_at}"
            )

            info_dialog = MDDialog(
                title="Company Information",
                text=info_text,
                buttons=[
                    MDRaisedButton(
                        text="CLOSE",
                        on_release=lambda x: info_dialog.dismiss(),
                        md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                    )
                ],
            )
            info_dialog.open()

        except Exception as e:
            self.show_error_dialog("Error Fetching Company Info", str(e))
            print(f"[DEBUG] Exception in show_company_info: {e}")  # Debugging

    def show_error_dialog(self, title, text):
        """Displays an error dialog."""
        dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: dialog.dismiss(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                )
            ],
        )
        dialog.open()

    def go_back(self):
        """Navigates back to the company list screen."""
        app = MDApp.get_running_app()
        app.sm.current = "notepad_screen"
        #print(exit)  # Debugging


class ProductListScreen(Screen):
    """Screen to display and manage products within a selected company."""

    def on_pre_enter(self, *args):
        super().on_pre_enter(*args)
        self.build_ui()
        self.load_products()

    def build_ui(self):
        """Builds the UI for the ProductListScreen."""
        layout = BoxLayout(orientation='vertical')

        # Toolbar with enhanced color
        toolbar = MDTopAppBar(
            title="Products",
            elevation=10,
            pos_hint={"top": 1},
            md_bg_color=MDApp.get_running_app().theme_cls.primary_color,
            left_action_items=[["arrow-left", lambda x: self.go_back()]],
            right_action_items=[["plus", lambda x: self.open_add_product_dialog()]]
        )
        layout.add_widget(toolbar)

        # RecycleView for product list
        self.product_list = ProductListView()
        layout.add_widget(self.product_list)

        # Spinner Overlay
        self.spinner = MDSpinner(
            size_hint=(None, None),
            size=(dp(46), dp(46)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            active=False
        )
        layout.add_widget(self.spinner)

        self.add_widget(layout)

    def load_products(self):
        """Loads products from Firebase and displays them."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        print(f"[DEBUG] Loading products for company '{company}' and user_id: {user_id}")  # Debugging

        if not user_id or not company:
            self.show_error_dialog("Selection Error", "No company selected.")
            print("[DEBUG] Selection Error: No company selected.")  # Debugging
            return

        # Activate spinner
        self.spinner.active = True

        # Start a new thread to fetch products
        threading.Thread(target=self.fetch_products, args=(user_id, company), daemon=True).start()

    def fetch_products(self, user_id, company):
        """Fetches products from Firebase in a separate thread."""
        app = MDApp.get_running_app()
        try:
            products = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").get().val()
            print(f"[DEBUG] Products fetched from Firebase: {products}")  # Debugging
            products_data = []

            if products:
                for name, details in products.items():
                    products_data.append({
                        "name": name,
                        "quantity": details.get("quantity", 0),
                        "price": details.get("price", 0.0),
                        "tax": details.get("tax", 0.0)
                    })
            else:
                products_data.append({
                    "name": "No products found. Add a new product.",
                    "quantity": 0,
                    "price": 0.0,
                    "tax": 0.0
                })

            # Schedule adding widgets to the UI thread
            Clock.schedule_once(lambda dt: self.update_product_list(products_data), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Loading Products", str(e)), 0)
            print(f"[DEBUG] Exception in fetch_products: {e}")  # Debugging
        finally:
            # Deactivate spinner on the UI thread
            Clock.schedule_once(lambda dt: setattr(self.spinner, 'active', False), 0)

    def update_product_list(self, products_data):
        """Updates the product list in the UI thread."""
        print("[DEBUG] Updating product list with RecycleView data.")
        if len(products_data) == 1 and products_data[0]["name"].startswith("No products found"):
            self.product_list.viewclass = 'NoProductItem'
            self.product_list.data = [
                {"message": products_data[0]["name"]}
            ]
        else:
            self.product_list.viewclass = 'ProductItem'
            self.product_list.data = [
                {
                    "name": product["name"],
                    "quantity": product["quantity"],
                    "price": product["price"],
                    "tax": product["tax"],
                    "is_dummy": False
                } for product in products_data
            ]
        self.product_list.refresh_from_data()  # Force refresh
        print("[DEBUG] Product list updated.")

    def open_add_product_dialog(self):
        """Opens a dialog to add a new product."""
        self.add_product_dialog = MDDialog(
            title="Add Product",
            type="custom",
            content_cls=AddProductContent(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: self.add_product_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="SAVE",
                    on_release=lambda x: self.save_product(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                ),
            ],
        )
        self.add_product_dialog.open()

    def save_product(self):
        """Saves the new product to Firebase."""
        name = self.add_product_dialog.content_cls.product_name.text.strip()
        quantity = self.add_product_dialog.content_cls.product_qty.text.strip()
        price = self.add_product_dialog.content_cls.product_price.text.strip()
        tax = self.add_product_dialog.content_cls.product_tax.text.strip()

        # Enhanced validation
        if not name:
            self.show_error_dialog("Validation Error", "Product name cannot be empty.")
            print("[DEBUG] Validation Error: Product name is empty.")
            return

        if not quantity.isdigit():
            self.show_error_dialog("Validation Error", "Quantity must be a whole number.")
            print("[DEBUG] Validation Error: Quantity is not a whole number.")
            return

        try:
            quantity_int = int(quantity)
            if quantity_int < 0:
                raise ValueError("Quantity cannot be negative.")
        except ValueError as ve:
            self.show_error_dialog("Validation Error", f"Invalid quantity: {ve}")
            print(f"[DEBUG] Validation Error: {ve}")
            return

        try:
            price_float = float(price)
            if price_float < 0:
                raise ValueError("Price cannot be negative.")
        except ValueError as ve:
            self.show_error_dialog("Validation Error", f"Invalid price: {ve}")
            print(f"[DEBUG] Validation Error: {ve}")
            return

        if tax:
            try:
                tax_float = float(tax)
                if tax_float < 0:
                    raise ValueError("Tax percentage cannot be negative.")
            except ValueError as ve:
                self.show_error_dialog("Validation Error", f"Invalid tax percentage: {ve}")
                print(f"[DEBUG] Validation Error: {ve}")
                return
        else:
            tax_float = 0.0  # Default tax if not provided

        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        print(
            f"[DEBUG] Saving product '{name}' with quantity {quantity_int}, price {price_float}, tax {tax_float} for company '{company}'")  # Debugging

        try:
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(name).set({
                    "quantity": quantity_int,
                    "price": price_float,
                    "tax": tax_float,
                    "created_at": int(time.time())
                })
            print(f"[DEBUG] Product '{name}' saved successfully with price and tax.")  # Debugging
            self.add_product_dialog.dismiss()
            self.load_products()
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Saving Product", str(e)), 0)
            print(f"[DEBUG] Exception in save_product: {e}")  # Debugging

    def select_product(self, product_name):
        """Selects a product and navigates to the transaction screen."""
        app = MDApp.get_running_app()
        app.sm.active_product = product_name
        print(f"[DEBUG] Active product set to: {product_name}")  # Debugging
        app.sm.current = "transaction_screen"

    def delete_product(self, product_name):
        """Deletes a product from Firebase and refreshes the list."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company

        # Confirmation Dialog
        confirm_dialog = MDDialog(
            title="Delete Product",
            text=f"Are you sure you want to delete '{product_name}'?",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: confirm_dialog.dismiss()
                ),
                MDRaisedButton(
                    text="DELETE",
                    on_release=lambda x: self.confirm_delete_product(confirm_dialog, company, product_name),
                    md_bg_color=MDApp.get_running_app().theme_cls.error_color
                ),
            ],
        )
        confirm_dialog.open()

    def confirm_delete_product(self, dialog, company, product_name):
        """Confirms and deletes the product."""
        dialog.dismiss()
        try:
            app = MDApp.get_running_app()
            user_id = app.user_id  # Correctly retrieve user_id
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(product_name).remove()
            print(f"[DEBUG] Product '{product_name}' deleted successfully.")
            self.load_products()
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Deleting Product", str(e)), 0)
            print(f"[DEBUG] Exception in delete_product: {e}")  # Debugging

    def show_product_info(self, product_name):
        """Displays detailed information about the selected product."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company

        try:
            # Fetch product details from Firebase
            product_data = app.db.child("users").child(user_id).child("companies") \
                                  .child(company).child("products").child(product_name).get().val()

            if not product_data:
                self.show_error_dialog("Error", "Product data not found.")
                print("[DEBUG] Product data not found.")  # Debugging
                return

            price = product_data.get("price", 0.0)
            tax = product_data.get("tax", 0.0)
            quantity = product_data.get("quantity", 0)
            total_stock_value = (price * quantity) + (price * quantity * tax / 100)
            created_at_ts = product_data.get("created_at", 0)
            created_at = datetime.fromtimestamp(created_at_ts).strftime('%Y-%m-%d %H:%M:%S')

            info_text = (
                f"Name: {product_name}\n"
                f"Quantity: {quantity}\n"
                f"Price per Unit: ${price:.2f}\n"
                f"Tax Percentage: {tax:.1f}%\n"
                f"Total Stock Value: ${total_stock_value:.2f}\n"
                f"Created At: {created_at}"
            )

            info_dialog = MDDialog(
                title="Product Information",
                text=info_text,
                buttons=[
                    MDRaisedButton(
                        text="CLOSE",
                        on_release=lambda x: info_dialog.dismiss(),
                        md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                    )
                ],
            )
            info_dialog.open()

        except Exception as e:
            self.show_error_dialog("Error Fetching Product Info", str(e))
            print(f"[DEBUG] Exception in show_product_info: {e}")  # Debugging

    def show_error_dialog(self, title, text):
        """Displays an error dialog."""
        dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: dialog.dismiss(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                )
            ],
        )
        dialog.open()

    def go_back(self):
        """Navigates back to the company list screen."""
        app = MDApp.get_running_app()
        app.sm.current = "company_list_screen"
        print("[DEBUG] Navigated back to CompanyListScreen")  # Debugging


class TransactionScreen(Screen):
    """Screen to handle purchases and sales for a selected product."""

    def on_pre_enter(self, *args):
        super().on_pre_enter(*args)
        self.build_ui()
        self.load_transactions()
        self.update_aggregator()

    def build_ui(self):
        """Builds the UI for the TransactionScreen."""
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))

        # Toolbar with enhanced color
        toolbar = MDTopAppBar(
            title="Transactions",
            elevation=10,
            pos_hint={"top": 1},
            left_action_items=[["arrow-left", lambda x: self.go_back()]],
            md_bg_color=MDApp.get_running_app().theme_cls.primary_color
        )
        layout.add_widget(toolbar)

        # Quantity Input and Buttons
        input_layout = BoxLayout(orientation='horizontal', spacing=dp(10), size_hint_y=None, height=dp(48))

        self.qty_field = MDTextField(
            hint_text="Quantity",
            helper_text="Enter amount",
            helper_text_mode="on_focus",
            input_filter='int',
            size_hint_x=0.5
        )
        purchase_button = MDRaisedButton(
            text="Purchase",
            md_bg_color=MDApp.get_running_app().theme_cls.primary_color,
            on_release=lambda x: self.record_purchase()
        )
        sale_button = MDRaisedButton(
            text="Sale",
            md_bg_color=MDApp.get_running_app().theme_cls.error_color,
            on_release=lambda x: self.record_sale()
        )

        input_layout.add_widget(self.qty_field)
        input_layout.add_widget(purchase_button)
        input_layout.add_widget(sale_button)

        layout.add_widget(input_layout)

        # Aggregator Label with enhanced styling
        self.aggregator_label = MDLabel(
            text="Monthly Totals...",
            halign="center",
            font_style="H6",
            size_hint_y=None,
            height=dp(48),
            theme_text_color="Primary",
            color=MDApp.get_running_app().theme_cls.primary_color  # Optional
        )
        layout.add_widget(self.aggregator_label)

        # RecycleView with transactions list
        self.transactions_list = TransactionListView()
        layout.add_widget(self.transactions_list)

        # Spinner Overlay
        self.spinner = MDSpinner(
            size_hint=(None, None),
            size=(dp(46), dp(46)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            active=False
        )
        layout.add_widget(self.spinner)

        self.add_widget(layout)

    def load_transactions(self):
        """Loads recent transactions (purchases and sales) from Firebase."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        product = app.sm.active_product
        print(
            f"[DEBUG] Loading transactions for user_id: {user_id}, company: {company}, product: {product}")  # Debugging

        if not user_id or not company or not product:
            self.show_error_dialog("Selection Error", "Incomplete selection.")
            print("[DEBUG] Selection Error: Incomplete selection.")  # Debugging
            return

        # Activate spinner
        self.spinner.active = True

        # Start a new thread to fetch transactions
        threading.Thread(target=self.fetch_transactions, args=(user_id, company, product), daemon=True).start()

    def fetch_transactions(self, user_id, company, product):
        """Fetches transactions from Firebase in a separate thread."""
        app = MDApp.get_running_app()
        try:
            # Fetch all purchases related to the company
            purchases = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("purchases").get().val()
            # Fetch all sales related to the company
            sales = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("sales").get().val()
            print(f"[DEBUG] Purchases fetched: {purchases}")  # Debugging
            print(f"[DEBUG] Sales fetched: {sales}")  # Debugging
            transactions_data = []

            # Filter purchases for the selected product
            if purchases:
                for purchase_id, purchase in purchases.items():
                    if purchase.get("product") == product:
                        date = datetime.fromtimestamp(purchase.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
                        transactions_data.append({
                            "text": f"Purchase - {purchase.get('product', '')}",
                            "secondary_text": f"Qty: {purchase.get('quantity', 0)} | Date: {date}"
                        })

            # Filter sales for the selected product
            if sales:
                for sale_id, sale in sales.items():
                    if sale.get("product") == product:
                        date = datetime.fromtimestamp(sale.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
                        transactions_data.append({
                            "text": f"Sale - {sale.get('product', '')}",
                            "secondary_text": f"Qty: {sale.get('quantity', 0)} | Date: {date}"
                        })

            # Sort transactions client-side by date descending
            transactions_data.sort(
                key=lambda x: datetime.strptime(x['secondary_text'].split('|')[1].strip().split('Date: ')[1],
                                                '%Y-%m-%d %H:%M:%S'), reverse=True)

            limited_transactions = transactions_data[:20]  # Limit to last 20 transactions

            # Schedule adding widgets to the UI thread
            Clock.schedule_once(lambda dt: self.update_transactions_list(limited_transactions), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Loading Transactions", str(e)), 0)
            print(f"[DEBUG] Exception in fetch_transactions: {e}")  # Debugging
        finally:
            # Deactivate spinner on the UI thread
            Clock.schedule_once(lambda dt: setattr(self.spinner, 'active', False), 0)

    def update_transactions_list(self, transactions_data):
        """Updates the transactions list in the UI thread."""
        print("[DEBUG] Updating transactions list with RecycleView data.")
        self.transactions_list.data = transactions_data
        self.transactions_list.refresh_from_data()  # Force refresh
        print("[DEBUG] Transactions list updated.")

    def record_purchase(self):
        """Records a purchase transaction."""
        quantity = self.qty_field.text.strip()
        if not quantity.isdigit():
            self.show_error_dialog("Validation Error", "Quantity must be a whole number.")
            print("[DEBUG] Validation Error: Quantity is not a whole number.")  # Debugging
            return

        quantity = int(quantity)
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        product = app.sm.active_product
        print(f"[DEBUG] Recording purchase: {quantity} of {product}")  # Debugging

        try:
            # Get current product data
            product_data = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(product).get().val()
            if not product_data:
                self.show_error_dialog("Error", "Product data not found.")
                print("[DEBUG] Product data not found.")  # Debugging
                return

            current_qty = product_data.get("quantity", 0)
            price = product_data.get("price", 0.0)
            tax = product_data.get("tax", 0.0)
            new_qty = current_qty + quantity
            total_price = price * quantity
            tax_amount = total_price * tax / 100
            total_cost = total_price + tax_amount

            print(f"[DEBUG] Current stock: {current_qty}, New stock after purchase: {new_qty}")  # Debugging
            print(
                f"[DEBUG] Total Price: {total_price}, Tax Amount: {tax_amount}, Total Cost: {total_cost}")  # Debugging

            # Update product quantity
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(product).update({"quantity": new_qty})
            print(f"[DEBUG] Updated product quantity in Firebase.")  # Debugging

            # Log the purchase
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("purchases").push({
                    "product": product,
                    "quantity": quantity,
                    "price_per_unit": price,
                    "tax_percentage": tax,
                    "total_cost": total_cost,
                    "timestamp": int(time.time())
                })
            print(f"[DEBUG] Purchase logged in Firebase.")  # Debugging

            # Show receipt
            self.show_receipt_dialog("Purchase", product, quantity, new_qty, total_cost)

            # Update UI
            self.qty_field.text = ""
            self.load_transactions()
            self.update_aggregator()

        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Recording Purchase", str(e)), 0)
            print(f"[DEBUG] Exception in record_purchase: {e}")  # Debugging

    def record_sale(self):
        """Records a sale transaction."""
        quantity = self.qty_field.text.strip()
        if not quantity.isdigit():
            self.show_error_dialog("Validation Error", "Quantity must be a whole number.")
            print("[DEBUG] Validation Error: Quantity is not a whole number.")  # Debugging
            return

        quantity = int(quantity)
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        product = app.sm.active_product
        print(f"[DEBUG] Recording sale: {quantity} of {product}")  # Debugging

        try:
            # Get current product data
            product_data = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(product).get().val()
            if not product_data:
                self.show_error_dialog("Error", "Product data not found.")
                print("[DEBUG] Product data not found.")  # Debugging
                return

            current_qty = product_data.get("quantity", 0)
            price = product_data.get("price", 0.0)
            tax = product_data.get("tax", 0.0)

            if quantity > current_qty:
                self.show_error_dialog("Insufficient Stock", "Not enough stock for sale.")
                print("[DEBUG] Insufficient stock for sale.")  # Debugging
                return

            new_qty = current_qty - quantity
            total_revenue = price * quantity
            tax_amount = total_revenue * tax / 100
            total_sale = total_revenue + tax_amount

            print(f"[DEBUG] Current stock: {current_qty}, New stock after sale: {new_qty}")  # Debugging
            print(
                f"[DEBUG] Total Revenue: {total_revenue}, Tax Amount: {tax_amount}, Total Sale: {total_sale}")  # Debugging

            # Update product quantity
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("products").child(product).update({"quantity": new_qty})
            print(f"[DEBUG] Updated product quantity in Firebase.")  # Debugging

            # Log the sale
            app.db.child("users").child(user_id).child("companies") \
                .child(company).child("sales").push({
                    "product": product,
                    "quantity": quantity,
                    "price_per_unit": price,
                    "tax_percentage": tax,
                    "total_sale": total_sale,
                    "timestamp": int(time.time())
                })
            print(f"[DEBUG] Sale logged in Firebase.")  # Debugging

            # Show receipt
            self.show_receipt_dialog("Sale", product, quantity, new_qty, total_sale)

            # Update UI
            self.qty_field.text = ""
            self.load_transactions()
            self.update_aggregator()

        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Recording Sale", str(e)), 0)
            print(f"[DEBUG] Exception in record_sale: {e}")  # Debugging

    def show_receipt_dialog(self, transaction_type, product_name, quantity, new_stock, total_amount):
        """Displays a receipt after a transaction."""
        receipt_text = (
            f"Transaction: {transaction_type}\n"
            f"Product: {product_name}\n"
            f"Quantity: {quantity}\n"
            f"New Stock: {new_stock}\n"
            f"Total Amount: ${total_amount:.2f}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        receipt_dialog = MDDialog(
            title="Receipt",
            text=receipt_text,
            buttons=[
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: receipt_dialog.dismiss(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                )
            ],
        )
        receipt_dialog.open()

    def update_aggregator(self):
        """Aggregates monthly purchases and sales."""
        user_id = MDApp.get_running_app().user_id
        app = MDApp.get_running_app()
        company = app.sm.active_company
        current_time = datetime.now()
        year = current_time.year
        month = current_time.month
        print(f"[DEBUG] Aggregating data for {year}-{month}")  # Debugging

        # Start a new thread to perform aggregation
        threading.Thread(target=self.perform_aggregation, args=(user_id, company, year, month), daemon=True).start()

    def perform_aggregation(self, user_id, company, year, month):
        """Performs data aggregation in a separate thread."""
        app = MDApp.get_running_app()
        try:
            # Fetch all purchases related to the company
            purchases = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("purchases").get().val()
            monthly_purchases = 0
            if purchases:
                for purchase in purchases.values():
                    ts = purchase.get("timestamp", 0)
                    dt = datetime.fromtimestamp(ts)
                    if dt.year == year and dt.month == month:
                        monthly_purchases += purchase.get("quantity", 0)
            print(f"[DEBUG] Monthly Purchases: {monthly_purchases}")  # Debugging

            # Fetch all sales related to the company
            sales = app.db.child("users").child(user_id).child("companies") \
                .child(company).child("sales").get().val()
            monthly_sales = 0
            if sales:
                for sale in sales.values():
                    ts = sale.get("timestamp", 0)
                    dt = datetime.fromtimestamp(ts)
                    if dt.year == year and dt.month == month:
                        monthly_sales += sale.get("quantity", 0)
            print(f"[DEBUG] Monthly Sales: {monthly_sales}")  # Debugging

            # Update the aggregator label on the UI thread
            aggregator_text = (
                f"Monthly Purchases (Qty): {monthly_purchases}\n"
                f"Monthly Sales (Qty): {monthly_sales}"
            )
            Clock.schedule_once(lambda dt: setattr(self.aggregator_label, 'text', aggregator_text), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt, e=e: self.show_error_dialog("Error Aggregating Data", str(e)), 0)
            print(f"[DEBUG] Exception in perform_aggregation: {e}")  # Debugging

    def go_back(self):
        """Navigates back to the product list screen."""
        app = MDApp.get_running_app()
        app.sm.current = "product_list_screen"

    def show_error_dialog(self, title, text):
        """Displays an error dialog."""
        dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: dialog.dismiss(),
                    md_bg_color=MDApp.get_running_app().theme_cls.primary_color
                )
            ],
        )
        dialog.open()
