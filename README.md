# FINTECH 533 Breakout Strategy

Develop, backtest, and document a breakout trading strategy using historical price data. The goal is to practice implementation, trade logging, performance evaluation, and clear communication of your methodology. Please present all results on your public GitHub site.

### Requirements

Your submitted GitHub page must contain **all** of the following:

1. Breakout Detection Function  
   - A clearly named and well-commented Python function (or class) that identifies breakouts.  
   - All parameters, thresholds, and cutoffs must be explicitly defined and easy to find (e.g., as named constants or function arguments at the top of the relevant code block).  
   - The same function must be explained in plain English on the web page (no code blocks on the page unless you choose to embed a small, readable snippet).

2. Trade Ledger & Blotter  (as usual)
A complete trade blotter (list of all trades) with:  
    Entry and exit timestamps (date or date-time)  
    Entry and exit prices  
    Position size (lot/share quantity)  
    Direction (long only, or long/short if you choose extend the strategy). Include this as both a downloadable CSV (or Parquet) file **and** a clean, readable table or Plotly visualization on your GitHub page. This is pretty easy to do with AI tools but if you hit a snag, ask for help!

3. Trade Outcome Analysis  
 A small histogram (or summary table) showing the fate of every backtested trade:  Successful (hit profit target or closed profitably), Timed out (market order to close after *n* days), or Stop-loss triggered. Clearly state your chosen timeout period and stop-loss logic in both the code and the write-up.

4. Performance Metrics
Report at minimum:  
    Average return per trade 
    Sharpe ratio (annualized, with risk-free rate assumption shown)  
    You are encouraged to include any additional relevant metrics (Sortino ratio, max drawdown, win rate, profit factor, expectancy, etc.)
    Display these metrics prominently on the web page with a short explanation of what each one tells us about the strategy.
    Recommended Implementation Approach

Pull **at least two years** of historical daily (or intraday) price data.  
Use a **rolling walk-forward window** of approximately one year for training/parameter optimization and the subsequent period for out-of-sample testing. This helps demonstrate robustness and reduces overfitting.  
You may use **LSEG Refinitiv** (via Datastream or Eikon API) **or** ShinyBroker for data retrieval.  
You are free to loop across multiple assets to identify one (or more) that exhibits frequent, tradable breakouts.
Website Content (GitHub Pages)
Your live page must include a well-written, professional section with the following:

Strategy Logic Paragraph: One clear paragraph describing the overall idea of your breakout strategy in plain English 

Asset Selection: State which asset(s) you chose and **why**. Explain your selection process (e.g., “I screened 50 futures contracts and selected CL (Crude Oil) because it showed the highest number of clean breakouts in the last two years…”). 
 
Breakout Definition: Clearly describe exactly how you define a breakout. Specify the technical features or indicators used (Donchian Channels, Bollinger Bands, price vs. recent high/low, volume confirmation, etc.) and all parameter values.

Tips for Success
- Keep the core logic simple and transparent — complexity can be added later as an extension.  
- Document any assumptions (slippage, commissions, data cleaning steps) so another student could replicate your results.  
- Use Plotly or Matplotlib for clean, professional charts (equity curve, drawdown, trade histogram).  
- Make sure your GitHub repo is public and the GitHub Pages site is live and mobile-friendly.

This assignment is deliberately structured so that a strong submission will also serve as an excellent portfolio piece for internships or quant trading interviews... and then we will make a better one for your final project!

If you have any questions about the requirements or want to run an early version of your code by me during office hours, please reach out — I’m happy to give quick feedback before the deadline.

Good luck, and happy coding!

