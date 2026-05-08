import random
import sys
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from game_states import (
    AUTOMA_CARDS,
    AutomaState,
    CAVE_CARDS,
    DRAGON_CARDS,
    GUILD_TILES,
    OBJECTIVE_TILES,
    SoloGameState,
)
from playout_compare import RNGOrder


class RNGSimulatorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wyrmspan RNG Simulator")

        self.game_state: Optional[SoloGameState] = None
        self.rng_order: Optional[RNGOrder] = None
        self.draw_history: List[Tuple[int, str, int, str]] = []
        self.draw_count = 0

        self._build_ui()
        self._update_ui_state(is_initialized=False)

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)

        layout.addWidget(self._build_setup_group())
        layout.addWidget(self._build_state_group())
        layout.addWidget(self._build_rng_group())
        layout.addWidget(self._build_history_group())

        self.setCentralWidget(container)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_setup_group(self) -> QGroupBox:
        group = QGroupBox("Game Setup")
        form = QFormLayout(group)

        seed_row = QHBoxLayout()
        self.seed_input = QLineEdit()
        self.seed_input.setValidator(QIntValidator(0, 2_147_483_647, self))
        self.seed_input.setPlaceholderText("Enter seed (integer)")
        self.seed_random_button = QPushButton("Randomize")
        self.seed_random_button.clicked.connect(self._randomize_seed)
        seed_row.addWidget(self.seed_input)
        seed_row.addWidget(self.seed_random_button)
        form.addRow("Seed", seed_row)

        self.difficulty_combo = QComboBox()
        for difficulty, name in sorted(AutomaState.difficulty_names.items()):
            self.difficulty_combo.addItem(name, difficulty)
        form.addRow("Automa Difficulty", self.difficulty_combo)

        button_row = QHBoxLayout()
        self.init_button = QPushButton("Initialize Game")
        self.init_button.clicked.connect(self._initialize_game)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_game)
        button_row.addWidget(self.init_button)
        button_row.addWidget(self.reset_button)
        form.addRow("", button_row)

        return group

    def _build_state_group(self) -> QGroupBox:
        group = QGroupBox("Starting State")
        layout = QVBoxLayout(group)
        self.state_text = QPlainTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMinimumHeight(160)
        layout.addWidget(self.state_text)
        return group

    def _build_rng_group(self) -> QGroupBox:
        group = QGroupBox("RNG Draw Control")
        layout = QHBoxLayout(group)

        deck_group = QGroupBox("Deck")
        deck_layout = QVBoxLayout(deck_group)
        self.cave_radio = QRadioButton("Caves")
        self.dragon_radio = QRadioButton("Dragons")
        self.automa_radio = QRadioButton("Automa")
        self.cave_radio.setChecked(True)
        self.deck_buttons = QButtonGroup(self)
        self.deck_buttons.addButton(self.cave_radio)
        self.deck_buttons.addButton(self.dragon_radio)
        self.deck_buttons.addButton(self.automa_radio)
        deck_layout.addWidget(self.cave_radio)
        deck_layout.addWidget(self.dragon_radio)
        deck_layout.addWidget(self.automa_radio)

        draw_group = QGroupBox("Draw")
        draw_layout = QFormLayout(draw_group)
        self.draw_count_input = QSpinBox()
        self.draw_count_input.setRange(1, 200)
        self.draw_count_input.setValue(1)
        self.draw_button = QPushButton("Draw")
        self.draw_button.clicked.connect(self._draw_cards)
        draw_layout.addRow("Times", self.draw_count_input)
        draw_layout.addRow("", self.draw_button)

        layout.addWidget(deck_group)
        layout.addWidget(draw_group)
        layout.addStretch(1)

        return group

    def _build_history_group(self) -> QGroupBox:
        group = QGroupBox("Draw History")
        layout = QVBoxLayout(group)

        filter_row = QHBoxLayout()
        filter_label = QLabel("Show")
        self.history_filter_combo = QComboBox()
        self.history_filter_combo.addItem("All", "all")
        self.history_filter_combo.addItem("Dragons", "dragon")
        self.history_filter_combo.addItem("Caves", "cave")
        self.history_filter_combo.addItem("Automa", "automa")
        self.history_filter_combo.currentIndexChanged.connect(self._refresh_history_table)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.history_filter_combo)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Order", "Deck", "Card ID", "Card Name"])
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)

        layout.addWidget(self.history_table)
        return group

    def _randomize_seed(self) -> None:
        seed_value = random.randint(0, 1_000_000_000)
        self.seed_input.setText(str(seed_value))

    def _initialize_game(self) -> None:
        seed_text = self.seed_input.text().strip()
        if not seed_text:
            QMessageBox.warning(self, "Missing Seed", "Please enter a seed value.")
            return
        try:
            seed_value = int(seed_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Seed", "Seed must be an integer.")
            return

        difficulty = int(self.difficulty_combo.currentData())
        random.seed(seed_value)
        self.game_state = SoloGameState(automa_difficulty=difficulty)
        self.game_state.create_game()
        self.rng_order = RNGOrder(self.game_state)
        self.draw_history.clear()
        self.draw_count = 0
        self._refresh_history_table()
        self._render_starting_state()
        self._update_ui_state(is_initialized=True)
        self._update_status()

    def _reset_game(self) -> None:
        self.game_state = None
        self.rng_order = None
        self.draw_history.clear()
        self.draw_count = 0
        self._refresh_history_table()
        self.state_text.clear()
        self._update_ui_state(is_initialized=False)
        self.status_bar.showMessage("Ready")

    def _draw_cards(self) -> None:
        if not self.rng_order:
            return
        if self.automa_radio.isChecked():
            deck_name = "automa"
        else:
            deck_name = "cave" if self.cave_radio.isChecked() else "dragon"
        draw_times = int(self.draw_count_input.value())
        remaining = self._remaining_in_deck(deck_name)
        if draw_times > remaining:
            QMessageBox.warning(
                self,
                "Deck Exhausted",
                f"Requested {draw_times} draws, but only {remaining} remain in the {deck_name} deck.",
            )
            return

        for _ in range(draw_times):
            card_id = self.rng_order._draw_from_deck(deck_name)
            if card_id is None:
                break
            self.draw_count += 1
            card_name = self._lookup_card_name(deck_name, card_id)
            self.draw_history.append((self.draw_count, deck_name, card_id, card_name))

        self._refresh_history_table()
        self._update_status()

    def _refresh_history_table(self) -> None:
        if not hasattr(self, "history_filter_combo"):
            return
        deck_filter = self.history_filter_combo.currentData() or "all"
        self.history_table.setRowCount(0)
        for order, deck_name, card_id, card_name in reversed(self.draw_history):
            if deck_filter != "all" and deck_name != deck_filter:
                continue
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(str(order)))
            self.history_table.setItem(row, 1, QTableWidgetItem(deck_name))
            self.history_table.setItem(row, 2, QTableWidgetItem(str(card_id)))
            self.history_table.setItem(row, 3, QTableWidgetItem(card_name))

    def _update_ui_state(self, is_initialized: bool) -> None:
        self.draw_button.setEnabled(is_initialized)
        self.draw_count_input.setEnabled(is_initialized)
        self.cave_radio.setEnabled(is_initialized)
        self.dragon_radio.setEnabled(is_initialized)
        self.automa_radio.setEnabled(is_initialized)
        self.reset_button.setEnabled(is_initialized)

        self.seed_input.setEnabled(not is_initialized)
        self.seed_random_button.setEnabled(not is_initialized)
        self.difficulty_combo.setEnabled(not is_initialized)
        self.init_button.setEnabled(not is_initialized)

    def _render_starting_state(self) -> None:
        if not self.game_state:
            self.state_text.clear()
            return

        player = self.game_state.player
        guild_idx = self.game_state.board["guild"]["guild_index"]
        guild_name = GUILD_TILES[guild_idx].get("name", f"Guild {guild_idx}")
        objectives = self.game_state.board["round_tracker"]["objectives"]
        objective_lines = []
        for idx, side in objectives:
            text = OBJECTIVE_TILES[idx][side].get("text", "")
            objective_lines.append(f"- Objective {idx} ({side}): {text}")

        dragon_display = self.game_state.board["card_display"]["dragon_cards"]
        cave_display = self.game_state.board["card_display"]["cave_cards"]

        state_lines = [
            f"Guild: {guild_name}",
            "Objectives:",
            *objective_lines,
            "",
            "Starting Dragon Hand:",
            *[f"- {card_id}: {self._lookup_card_name('dragon', card_id)}" for card_id in player.dragon_hand],
            "",
            "Starting Cave Hand:",
            *[f"- {card_id}: {self._lookup_card_name('cave', card_id)}" for card_id in player.cave_hand],
            "",
            "Card Display (Dragons):",
            *[f"- {card_id}: {self._lookup_card_name('dragon', card_id)}" for card_id in dragon_display],
            "",
            "Card Display (Caves):",
            *[f"- {card_id}: {self._lookup_card_name('cave', card_id)}" for card_id in cave_display],
        ]

        self.state_text.setPlainText("\n".join(state_lines))

    def _lookup_card_name(self, deck_name: str, card_id: int) -> str:
        if deck_name == "dragon":
            if 0 <= card_id < len(DRAGON_CARDS):
                return DRAGON_CARDS[card_id].get("name", f"Dragon {card_id}")
            return f"Dragon {card_id}"
        if deck_name == "automa":
            if 0 <= card_id < len(AUTOMA_CARDS):
                card = AUTOMA_CARDS[card_id]
                corner = card.get("corner_id", "?")
                pass_flag = card.get("pass", False)
                refresh_flag = card.get("refresh", False)
                adjust = card.get("adjust_objective", 0)
                return f"Corner {corner}, pass={pass_flag}, refresh={refresh_flag}, adjust={adjust}"
            return f"Automa {card_id}"
        if 0 <= card_id < len(CAVE_CARDS):
            return CAVE_CARDS[card_id].get("text", f"Cave {card_id}")
        return f"Cave {card_id}"

    def _remaining_in_deck(self, deck_name: str) -> int:
        if not self.rng_order:
            return 0
        if deck_name == "dragon":
            return self.rng_order._dragon_idx + 1
        if deck_name == "cave":
            return self.rng_order._cave_idx + 1
        if deck_name == "automa":
            return self.rng_order._automa_idx + 1
        return 0

    def _update_status(self) -> None:
        if not self.rng_order:
            return
        dragon_remaining = self._remaining_in_deck("dragon")
        cave_remaining = self._remaining_in_deck("cave")
        automa_remaining = self._remaining_in_deck("automa")
        self.status_bar.showMessage(
            f"Remaining - Dragons: {dragon_remaining}, Caves: {cave_remaining}, Automa: {automa_remaining}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = RNGSimulatorWindow()
    window.resize(980, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
