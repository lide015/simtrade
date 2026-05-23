-- L1 raw market candles, keyed (symbol, timeframe, ts)
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    closed     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv(symbol, timeframe, ts);

-- L6 main table: one row per decision
CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    market_snapshot TEXT NOT NULL,
    trader_state    TEXT NOT NULL,
    action          TEXT NOT NULL,
    post_outcome    TEXT,
    is_locked       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_setup
    ON decisions(json_extract(trader_state, '$.reasoning_tags.setup_type'));
CREATE INDEX IF NOT EXISTS idx_regime
    ON decisions(json_extract(trader_state, '$.reasoning_tags.market_regime'));

-- L7 experiments: hypotheses proposed and pending validation
CREATE TABLE IF NOT EXISTS experiments (
    id                   TEXT PRIMARY KEY,
    proposed_at          TEXT NOT NULL,
    hypothesis           TEXT NOT NULL,
    conditions           TEXT NOT NULL,
    target_n             INTEGER NOT NULL,
    status               TEXT NOT NULL DEFAULT 'proposed',
    conclusion           TEXT,
    related_decision_ids TEXT
);

-- L7 self-evaluation log: did each discovery actually hold up?
CREATE TABLE IF NOT EXISTS discovery_log (
    id                  TEXT PRIMARY KEY,
    detected_at         TEXT NOT NULL,
    type                TEXT NOT NULL,
    pattern_description TEXT NOT NULL,
    p_value             REAL,
    effect_size         REAL,
    n_samples           INTEGER,
    hit_or_miss         INTEGER
);

-- L4 closed positions (used for L5 stats and L6 outcome reconciliation)
CREATE TABLE IF NOT EXISTS positions (
    id           TEXT PRIMARY KEY,
    decision_id  TEXT,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    size         REAL NOT NULL,
    entry_price  REAL NOT NULL,
    entry_ts     TEXT NOT NULL,
    exit_price   REAL,
    exit_ts      TEXT,
    exit_reason  TEXT,
    realized_pnl REAL,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_pos_decision ON positions(decision_id);
