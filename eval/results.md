# Evaluation results

70 questions in `eval/questions.jsonl`: 60 answerable, 10 not covered by the help center. The help center has 21 articles split into 68 sections.

## Retrieval

| Retriever | Recall@1 | Recall@3 | MRR | Right article in top 5 sections |
|---|---|---|---|---|
| bm25-nostem | 50% | 73% | 0.65 | 78% |
| bm25 | 57% | 75% | 0.69 | 83% |
| embeddings | 58% | 82% | 0.72 | 83% |
| hybrid | 70% | 83% | 0.79 | 88% |

### Questions where the right article wasn't ranked first

**bm25-nostem** (30)

- q01: "How much does shipping cost?" ranked `damaged-or-wrong-items` first; right article at rank 5
- q02: "My cart is $82 after my coupon. Do I still have to pay for shipping?" ranked `account-and-emails` first; right article at rank 5
- q03: "If I pick the cheapest shipping option, how many days until my package gets here?" ranked `lost-or-delayed-packages` first; right article at rank 4
- q04: "I need my hoodie by this weekend. Can I pay to get it faster?" ranked `payment-methods` first; right article at rank 3
- q05: "I ordered at 4 in the afternoon. Will it go out today?" ranked `returns` first; right article at rank 20
- q06: "Can you mail an order to an APO address for my brother overseas?" ranked `changing-or-canceling-an-order` first; right article at rank 2
- q07: "Do you deliver to Toronto?" ranked `account-and-emails` first; right article at rank 10
- q10: "The tracking link shows nothing yet even though I got the shipping email this morning." ranked `lost-or-delayed-packages` first; right article at rank 2
- q12: "My package has been stuck in the same city for over a week." ranked `gift-cards` first; right article at rank 6
- q13: "Is it okay to return a shirt I wore once and washed?" ranked `changing-or-canceling-an-order` first; right article at rank 3
- q14: "How long do I have to send something back?" ranked `damaged-or-wrong-items` first; right article at rank 18
- q15: "Can I return the pin set if I changed my mind?" ranked `hats-and-accessories` first; right article at rank 2
- q17: "The medium tee is too tight. Can I trade it for a large?" ranked `size-guide-tops` first; right article at rank 9
- q20: "Will I get my shipping cost back when I return something?" ranked `damaged-or-wrong-items` first; right article at rank 5
- q21: "My hat showed up with the strap ripped." ranked `hats-and-accessories` first; right article at rank 6
- q22: "You sent me a crewneck but I ordered a hoodie." ranked `size-guide-tops` first; right article at rank 2
- q23: "The print on my tee started peeling after two months." ranked `size-guide-tops` first; right article at rank 3
- q26: "Will the t-shirt shrink if I dry it?" ranked `washing-and-care` first; right article at rank 2
- q28: "I have a big head. Will the cap fit me?" ranked `size-guide-tops` first; right article at rank 10
- q31: "How should I wash my hoodie so the design doesn't crack?" ranked `size-guide-tops` first; right article at rank 2
- q32: "Can I throw the dad hat in the washer?" ranked `hats-and-accessories` first; right article at rank 2
- q35: "Is there a sale going on right now?" ranked `returns` first; right article at rank 8
- q41: "I put the wrong street address on my order. Can I fix it?" ranked `damaged-or-wrong-items` first; right article at rank 2
- q47: "The hoodie in my size is sold out. Will you get more?" ranked `exchanges` first; right article at rank 2
- q50: "Our club wants 30 matching hoodies. Do you give a discount?" ranked `damaged-or-wrong-items` first; right article at rank 11
- q51: "Can you print our team logo on your shirts?" ranked `contact-us` first; right article at rank 8
- q53: "I live in Anaheim. Can I just grab my order instead of paying shipping?" ranked `shipping-options` first; right article at rank 2
- q56: "How do I get in touch with a real person?" ranked `account-and-emails` first; right article at rank 4
- q57: "Is there a number I can call?" ranked `order-tracking` first; right article at rank 3
- q58: "How fast do you answer emails?" ranked `account-and-emails` first; right article at rank 5

**bm25** (26)

- q02: "My cart is $82 after my coupon. Do I still have to pay for shipping?" ranked `account-and-emails` first; right article at rank 5
- q04: "I need my hoodie by this weekend. Can I pay to get it faster?" ranked `payment-methods` first; right article at rank 3
- q05: "I ordered at 4 in the afternoon. Will it go out today?" ranked `returns` first; right article at rank 17
- q07: "Do you deliver to Toronto?" ranked `order-tracking` first; right article at rank 11
- q08: "Will I get hit with customs fees if I order from Germany?" ranked `bulk-and-wholesale` first; right article at rank 2
- q12: "My package has been stuck in the same city for over a week." ranked `bulk-and-wholesale` first; right article at rank 5
- q13: "Is it okay to return a shirt I wore once and washed?" ranked `exchanges` first; right article at rank 5
- q14: "How long do I have to send something back?" ranked `damaged-or-wrong-items` first; right article at rank 18
- q15: "Can I return the pin set if I changed my mind?" ranked `hats-and-accessories` first; right article at rank 2
- q17: "The medium tee is too tight. Can I trade it for a large?" ranked `size-guide-tops` first; right article at rank 10
- q19: "I mailed my return last week. When will I see the money?" ranked `payment-methods` first; right article at rank 2
- q20: "Will I get my shipping cost back when I return something?" ranked `returns` first; right article at rank 6
- q21: "My hat showed up with the strap ripped." ranked `hats-and-accessories` first; right article at rank 3
- q22: "You sent me a crewneck but I ordered a hoodie." ranked `size-guide-tops` first; right article at rank 11
- q26: "Will the t-shirt shrink if I dry it?" ranked `washing-and-care` first; right article at rank 2
- q32: "Can I throw the dad hat in the washer?" ranked `hats-and-accessories` first; right article at rank 2
- q35: "Is there a sale going on right now?" ranked `returns` first; right article at rank 10
- q43: "Can I add another tee to the order I placed this morning?" ranked `discount-codes` first; right article at rank 2
- q47: "The hoodie in my size is sold out. Will you get more?" ranked `exchanges` first; right article at rank 2
- q50: "Our club wants 30 matching hoodies. Do you give a discount?" ranked `discount-codes` first; right article at rank 14
- q51: "Can you print our team logo on your shirts?" ranked `washing-and-care` first; right article at rank 4
- q53: "I live in Anaheim. Can I just grab my order instead of paying shipping?" ranked `payment-methods` first; right article at rank 4
- q56: "How do I get in touch with a real person?" ranked `privacy` first; right article at rank 5
- q57: "Is there a number I can call?" ranked `order-tracking` first; right article at rank 4
- q58: "How fast do you answer emails?" ranked `privacy` first; right article at rank 2
- q60: "Do you sell customer info to advertisers?" ranked `bulk-and-wholesale` first; right article at rank 3

**embeddings** (25)

- q04: "I need my hoodie by this weekend. Can I pay to get it faster?" ranked `payment-methods` first; right article at rank 9
- q05: "I ordered at 4 in the afternoon. Will it go out today?" ranked `contact-us` first; right article at rank 2
- q07: "Do you deliver to Toronto?" ranked `local-pickup-and-events` first; right article at rank 2
- q13: "Is it okay to return a shirt I wore once and washed?" ranked `washing-and-care` first; right article at rank 4
- q14: "How long do I have to send something back?" ranked `order-tracking` first; right article at rank 10
- q15: "Can I return the pin set if I changed my mind?" ranked `changing-or-canceling-an-order` first; right article at rank 6
- q17: "The medium tee is too tight. Can I trade it for a large?" ranked `washing-and-care` first; right article at rank 3
- q19: "I mailed my return last week. When will I see the money?" ranked `order-tracking` first; right article at rank 3
- q20: "Will I get my shipping cost back when I return something?" ranked `returns` first; right article at rank 6
- q21: "My hat showed up with the strap ripped." ranked `hats-and-accessories` first; right article at rank 4
- q22: "You sent me a crewneck but I ordered a hoodie." ranked `size-guide-tops` first; right article at rank 7
- q23: "The print on my tee started peeling after two months." ranked `washing-and-care` first; right article at rank 2
- q24: "I'm between a small and a medium in the tee. Which should I get?" ranked `washing-and-care` first; right article at rank 2
- q26: "Will the t-shirt shrink if I dry it?" ranked `washing-and-care` first; right article at rank 2
- q28: "I have a big head. Will the cap fit me?" ranked `size-guide-tops` first; right article at rank 3
- q34: "How do I get 10% off my first purchase?" ranked `returns` first; right article at rank 3
- q35: "Is there a sale going on right now?" ranked `shipping-options` first; right article at rank 3
- q39: "Can I split my payment into smaller chunks?" ranked `exchanges` first; right article at rank 4
- q43: "Can I add another tee to the order I placed this morning?" ranked `order-tracking` first; right article at rank 2
- q47: "The hoodie in my size is sold out. Will you get more?" ranked `size-guide-tops` first; right article at rank 3
- q50: "Our club wants 30 matching hoodies. Do you give a discount?" ranked `discount-codes` first; right article at rank 13
- q51: "Can you print our team logo on your shirts?" ranked `size-guide-tops` first; right article at rank 4
- q56: "How do I get in touch with a real person?" ranked `account-and-emails` first; right article at rank 2
- q57: "Is there a number I can call?" ranked `privacy` first; right article at rank 2
- q58: "How fast do you answer emails?" ranked `account-and-emails` first; right article at rank 5

**hybrid** (18)

- q04: "I need my hoodie by this weekend. Can I pay to get it faster?" ranked `payment-methods` first; right article at rank 2
- q05: "I ordered at 4 in the afternoon. Will it go out today?" ranked `changing-or-canceling-an-order` first; right article at rank 6
- q07: "Do you deliver to Toronto?" ranked `bulk-and-wholesale` first; right article at rank 4
- q13: "Is it okay to return a shirt I wore once and washed?" ranked `washing-and-care` first; right article at rank 4
- q14: "How long do I have to send something back?" ranked `damaged-or-wrong-items` first; right article at rank 17
- q15: "Can I return the pin set if I changed my mind?" ranked `changing-or-canceling-an-order` first; right article at rank 2
- q17: "The medium tee is too tight. Can I trade it for a large?" ranked `size-guide-tops` first; right article at rank 6
- q20: "Will I get my shipping cost back when I return something?" ranked `returns` first; right article at rank 4
- q21: "My hat showed up with the strap ripped." ranked `hats-and-accessories` first; right article at rank 3
- q22: "You sent me a crewneck but I ordered a hoodie." ranked `size-guide-tops` first; right article at rank 5
- q26: "Will the t-shirt shrink if I dry it?" ranked `washing-and-care` first; right article at rank 2
- q35: "Is there a sale going on right now?" ranked `returns` first; right article at rank 5
- q47: "The hoodie in my size is sold out. Will you get more?" ranked `exchanges` first; right article at rank 3
- q50: "Our club wants 30 matching hoodies. Do you give a discount?" ranked `discount-codes` first; right article at rank 12
- q51: "Can you print our team logo on your shirts?" ranked `washing-and-care` first; right article at rank 2
- q53: "I live in Anaheim. Can I just grab my order instead of paying shipping?" ranked `payment-methods` first; right article at rank 2
- q56: "How do I get in touch with a real person?" ranked `account-and-emails` first; right article at rank 3
- q58: "How fast do you answer emails?" ranked `privacy` first; right article at rank 4
