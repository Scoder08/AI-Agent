from datetime import datetime
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from utils.llmUtils import getLLM
from tools.tools_list import tools_list
from utils.datetime_utils import get_current_time_with_offset
from utils.llmUtils import filter_messages
tool_node = ToolNode(tools_list["SATWIK"])



table_name_structure_map = {
    "analytics.order_items_view": """
        CREATE TABLE analytics.order_items_view
        (
            `order_item_id` UInt64,
            `created_at` DateTime64(3),
            `order_id` UInt64,
            `order_item_type` LowCardinality(String),
            `order_item_name` String,
            `order_date` DateTime,
            `user_id` UInt64,
            `payment_method` LowCardinality(String),
            `order_source` LowCardinality(String),
            `order_status` String,
            `product_id` UInt64,
            `variation_id` UInt64,
            `quantity` UInt32,
            `price_after_coupon_and_wallet` Decimal(9, 2),
            `asp` Decimal(9, 2),
            `mrp` Decimal(9, 2),
            `item_status` LowCardinality(String),
            `cancellation_requested_by_admin` LowCardinality(String),
            `cancellation_request_time` DateTime,
            `cancelled_reason` LowCardinality(String),
            `cancellation_comment` String,
            `delivered_time` DateTime,
            `first_edd` DateTime,
            `cancelled_time` DateTime,
            `dispatched_time` DateTime,
            `return_initiated_time` DateTime,
            `return_approved_observed_time` DateTime,
            `fullfillable_time` DateTime,
            `returned_time` DateTime,
            `dto_return` Bool,
            `rto_return` Bool,
            `exchanged_order_item_id` UInt64 DEFAULT 0,
            `bogo_sale_item` Bool,
            `return_picked_up_time` DateTime,
            `return_reason` String,
            `return_comment` String,
            `return_issue` String,
            `return_by_admin` String,
            `return_request_time` DateTime,
            `return_reverse_pickup_failed_time` DateTime,
            `shipping_provider` String,
            `dispatch_facility` String,
            `return_admin_comment` String,
            `return_shipping_provider` String,
            `return_canceled_by_customer` String,
            `courier_assignment_failure_count` UInt8,
            `customer_cancelled_return_reverse_pickup_time` DateTime,
            `inventory_type` LowCardinality(String),
            `tracking_number` String,
            `return_tracking_number` String,
            `utm_source` String,
            `utm_medium` String,
            `utm_campaign` String,
            `city` String,
            `state` String,
            `pincode` String,
            `coupon_name` String,
            `coupon_discount` Int16,
            `shipping_or_cod_charge` UInt16,
            `normal_wallet_used` UInt16,
            `fast_wallet_used` UInt16,
            `bogo_discount_used` UInt16,
            `parent_order_item_id` UInt64 DEFAULT 0,
            `metadata` String DEFAULT '',
            `billing_address` String DEFAULT '',
            `payment_gateway` LowCardinality(String) DEFAULT '',
            `invoice_date` DateTime,
            `invoice_code` String DEFAULT '',
            `invoice_amount` Decimal(9, 2),
            `refund_cod_amount` Decimal(9, 2),
            `refund_payout_link_send_time` DateTime,
            `rp_payout_link_status` String DEFAULT '',
            `rp_payout_link_attempts` UInt8,
            `payout_link_processed_time` DateTime,
            `vpa_refund_payout_id` String,
            `vpa_refund_failure_reason` String,
            `vpa_refund_payout_time` DateTime,
            `vpa_refund_payout_status` String,
            `vpa_refund_payout_attempt_time` DateTime,
            `refund_initiated_date` DateTime,
            `refund_completed_date` DateTime,
            `manual_refund` String,
            `manual_refund_time` DateTime,
            `not_picked_up_but_refunded` String,
            `refund_status` String,
            `refund_amount` Decimal(9, 2),
            `normal_wallet_refund_amount` Decimal(9, 2),
            `fast_wallet_refund_amount` Decimal(9, 2),
            `return_images` String,
            `onhold_reason` LowCardinality(String),
            `qc_failed_return_rejected` Bool,
            `billing_phone` String,
            `courier_partner_edd` String,
            `rvp_escalated_time` DateTime,
            `last_rvp_cancelled_time` DateTime,
            `store_order_remarks` String,
            `shipping_alternate_phone` String,
            `fast_delivery_type` LowCardinality(String),
            `return_initiated_observed_time` DateTime,
            `payment_date` DateTime,
            `user_os` LowCardinality(String),
            `prepaid_shipping_fee` UInt16,
            `cod_shipping_fee` UInt16,
            `cod_charge_fee` UInt16,
            `is_freebie` LowCardinality(String),
            `return_window` Int64 DEFAULT -1,
            `billing_email` String,
            `return_disable_reason` String,
            `payment_method_type` LowCardinality(String),
            `payment_info` String,
            `offline_invoice_number` String,
            `return_refund_completed` LowCardinality(String),
            `refund_eligibility` LowCardinality(String),
            `reverse_pickup_failed_reason` String,
            `rk` UInt64,
            `fulfilled_facility_uc` LowCardinality(String),
            `fulfillable_time_uc_ist` DateTime,
            INDEX dispatched_time_idx dispatched_time TYPE minmax GRANULARITY 1,
            INDEX delivered_time_idx delivered_time TYPE minmax GRANULARITY 1,
            INDEX return_initiated_time_idx return_initiated_time TYPE minmax GRANULARITY 1,
            INDEX cancelled_time_idx cancelled_time TYPE minmax GRANULARITY 1,
            INDEX returned_time_idx returned_time TYPE minmax GRANULARITY 1,
            INDEX user_id_idx user_id TYPE minmax GRANULARITY 1,
            INDEX order_status_idx order_status TYPE minmax GRANULARITY 1,
            INDEX order_date_idx order_date TYPE minmax GRANULARITY 1
        )
        ENGINE = MergeTree
        PARTITION BY toYYYYMM(order_date)
        ORDER BY order_item_id
        SETTINGS index_granularity = 8192
        """,
}
# Set up the model with an instruction prompt
COLUMNS = [
    "order_item_id",
    "created_at",
    "order_id",
    "order_item_type",
    "order_item_name",
    "order_date",
    "user_id",
    "payment_method",
    "order_source",
    "order_status",
    "product_id",
    "variation_id",
    "quantity",
    "price_after_coupon_and_wallet",
    "asp",
    "mrp",
    "item_status",
    "cancellation_requested_by_admin",
    "cancellation_request_time",
    "cancelled_reason",
    "cancellation_comment",
    "delivered_time",
    "first_edd",
    "cancelled_time",
    "dispatched_time",
    "return_initiated_time",
    "return_approved_observed_time",
    "fullfillable_time",
    "returned_time",
    "dto_return",
    "rto_return",
    "exchanged_order_item_id",
    "bogo_sale_item",
    "return_picked_up_time",
    "return_reason",
    "return_comment",
    "return_issue",
    "return_by_admin",
    "return_request_time",
    "return_reverse_pickup_failed_time",
    "shipping_provider",
    "dispatch_facility",
    "return_admin_comment",
    "return_shipping_provider",
    "return_canceled_by_customer",
    "courier_assignment_failure_count",
    "customer_cancelled_return_reverse_pickup_time",
    "inventory_type",
    "tracking_number",
    "return_tracking_number",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "city",
    "state",
    "pincode",
    "coupon_name",
    "coupon_discount",
    "shipping_or_cod_charge",
    "normal_wallet_used",
    "fast_wallet_used",
    "bogo_discount_used",
    "parent_order_item_id",
    "metadata",
    "billing_address",
    "payment_gateway",
    "invoice_date",
    "invoice_code",
    "invoice_amount",
    "refund_cod_amount",
    "refund_payout_link_send_time",
    "rp_payout_link_status",
    "rp_payout_link_attempts",
    "payout_link_processed_time",
    "vpa_refund_payout_id",
    "vpa_refund_failure_reason",
    "vpa_refund_payout_time",
    "vpa_refund_payout_status",
    "vpa_refund_payout_attempt_time",
    "refund_initiated_date",
    "refund_completed_date",
    "manual_refund",
    "manual_refund_time",
    "not_picked_up_but_refunded",
    "refund_status",
    "refund_amount",
    "normal_wallet_refund_amount",
    "fast_wallet_refund_amount",
    "return_images",
    "onhold_reason",
    "qc_failed_return_rejected",
    "billing_phone",
    "courier_partner_edd",
    "rvp_escalated_time",
    "last_rvp_cancelled_time",
    "store_order_remarks",
    "shipping_alternate_phone",
    "fast_delivery_type",
    "return_initiated_observed_time",
    "payment_date",
    "user_os",
    "prepaid_shipping_fee",
    "cod_shipping_fee",
    "cod_charge_fee",
    "is_freebie",
    "return_window",
    "billing_email",
    "return_disable_reason",
    "payment_method_type",
    "payment_info",
    "offline_invoice_number",
    "return_refund_completed",
    "refund_eligibility",
    "reverse_pickup_failed_reason",
    "rk",
    "fulfilled_facility_uc",
    "fulfillable_time_uc_ist"
]
SCHEMA_SUMMARY = "analytics.order_items_view → " + ", ".join(COLUMNS)

instruction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""

You are **SATWIK**, an expert ClickHouse-SQL assistant.

**Rules (always obey)**  
1. Use **only** columns appearing in the schema summary below – never invent new ones.  
2. Cancellations ⇒ `cancelled_time` (≠ 'status' strings).  
3. Returns ⇒ `returned_time` or `return_initiated_time`.  
4. Refunds ⇒ `refund_status` / `refund_amount` / `refund_completed_date`.  
5. Orders placed ⇒ use `order_date`.  
6. Exclude `order_status` in ('wc-failed','trash','wc-pending') by default.  
7. Treat default nulls: DateTime '2000-01-01', UInt64 0, String ''.  
8. Format output **only** as a ```sql code-block (no explanation unless asked).  
9. Ask clarifying questions only if absolutely needed.
10. schema follows format `table_name → columns`
11. In query if comparing dates use date() in front, similarly time(), hour()
12. all possible item statuses : [wc-cancelled,wc-processing (it means it is under processing),wc-returned,wc-delivered,wc-return-initiated,wc-return-reverse-pickup-failed,wc-return-picked-up,wc-completed (it means dispatched),wc-awaiting-dispatch,wc-cancellation-requested,cancelled,wc-customer-cancelled-return-reverse-pickup,wc-return-rejected,wc-return-approved,wc-return-courier-assignment-failed,wc-return-requested]

ALWAYS prioritize schema compliance, correctness, and clean formatting. You are writing for production analytics use cases.
**ALWAYS STICK TO THIS SCHEMA , Schema : **  
{SCHEMA_SUMMARY}
"""
    ),
    ("placeholder", "{messages}")
])

# Compose prompt with LLM
llm = getLLM("openai", "SATWIK")
model = instruction_prompt | llm  # ✅ No bind_tools here unless you have actual tool calling logic

# Assistant node logic

async def invoke(state: dict, config: RunnableConfig):
    while True:
        state["messages"] = filter_messages(state["messages"])
        state["messages"].append(("system", f"Today is {get_current_time_with_offset(config).strftime('%d-%m-%Y %H:%M:%S')}. TimeZone configured for the logged in user in the Mobile Cloud is {config.get('configurable', {}).get('timezone', 'Asia/Kolkata')}, with an offset of {config.get('configurable', {}).get('timezone_offset', 330)} minutes."))
        result = await model.ainvoke(state, config=config)
        if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
            state["messages"].append(("user", "Please provide a meaningful response."))
        else:
            break
    return {"messages": result}

# Conditional edge: whether to continue
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    return "continue" if getattr(last_message, "tool_calls", None) else "end"

# LangGraph setup
memory = MemorySaver()
graph = StateGraph(MessagesState)

graph.add_node("assistant", invoke)
graph.add_node("tools", tool_node)

graph.add_edge(START, "assistant")
graph.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "assistant")

# Compile final SATWIK agent
satwik = graph.compile(checkpointer=memory)