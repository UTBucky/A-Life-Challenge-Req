import pygame
import pytest
from environment import Environment
from button import Button, create_stop_start_button, create_save_button, create_load_button, create_skip_button
from viewer2dp import Viewer2D

pygame.init()


@pytest.fixture
def dummy_screen():
    return pygame.Surface((400, 400))


@pytest.fixture
def dummy_font():
    return pygame.font.SysFont(None, 36)


@pytest.fixture
def button(dummy_screen):
    return Button(
        screen=dummy_screen,
        font=pygame.font.SysFont(None, 36),
        text="Start"
    )


@pytest.fixture
def dummy_env():
    DummyEnvironment = Environment(50, 50)
    return DummyEnvironment


@pytest.fixture
def viewer(dummy_env):
    return Viewer2D(dummy_env)


def test_button_draw_no_crash(button):
    # Just checks that draw_button runs without crashing
    button.draw_button()


def test_create_stop_start_button_returns_button(dummy_screen, dummy_font):
    btn = create_stop_start_button(dummy_screen, dummy_font, True)
    assert isinstance(btn, Button)


def test_create_save_button_returns_button(dummy_screen, dummy_font):
    btn = create_save_button(dummy_screen, dummy_font)
    assert isinstance(btn, Button)


def test_create_load_button_returns_button(dummy_screen, dummy_font):
    btn = create_load_button(dummy_screen, dummy_font)
    assert isinstance(btn, Button)


def test_create_skip_button_returns_button(dummy_screen, dummy_font):
    btn = create_skip_button(dummy_screen, dummy_font)
    assert isinstance(btn, Button)


def test_handle_events_toggle_running(viewer):
    # Tests that start/stop button correctly sets _running attribute to true/false
    assert viewer._running is True

    # Simulate a mouse click inside the stop/start button
    rect = viewer._stop_start_button.get_rectangle()
    center_pos = (rect.x + 5, rect.y + 5)

    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": center_pos}))
    viewer.handle_events()
    assert viewer._running is False

    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": center_pos}))
    viewer.handle_events()
    assert viewer._running is True
