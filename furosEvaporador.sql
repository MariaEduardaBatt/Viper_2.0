DROP TABLE IF EXISTS furosEvaporador;

CREATE TABLE furosEvaporador (
    id INTEGER PRIMARY KEY,
    posicao_x INTEGER NOT NULL,
    posicao_y INTEGER NOT NULL,
    status_limpeza INTEGER NOT NULL DEFAULT 0
);

INSERT INTO furosEvaporador (id, posicao_x, posicao_y, status_limpeza) VALUES (1, 259, 95, 0);
INSERT INTO furosEvaporador (id, posicao_x, posicao_y, status_limpeza) VALUES (2, 333, 165, 0);
INSERT INTO furosEvaporador (id, posicao_x, posicao_y, status_limpeza) VALUES (3, 269, 301, 0);
INSERT INTO furosEvaporador (id, posicao_x, posicao_y, status_limpeza) VALUES (4, 481, 416, 0);
