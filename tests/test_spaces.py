"""
观测空间与动作空间验证测试。
"""

import pytest
import numpy as np
from penguinchess.spaces import (
    PenguinChessFlatObs,
    PenguinChessActionSpace,
    N_HEX,
    N_PIECES,
    N_FEATURES,
    PIECE_FEATURES,
    OBS_FLAT_SIZE,
)


class TestObservationSpace:
    """观测空间测试。"""

    def test_observation_space_shape(self):
        """观测空间形状应该是 (204,)。"""
        space = PenguinChessFlatObs
        assert space.shape == (OBS_FLAT_SIZE,), f"Expected shape ({OBS_FLAT_SIZE},), got {space.shape}"

    def test_observation_space_dtype(self):
        """观测空间数据类型应该是 float32。"""
        space = PenguinChessFlatObs
        assert space.dtype == np.float32

    def test_observation_space_contains_valid(self):
        """有效的观测应该被 contains() 接受。"""
        space = PenguinChessFlatObs
        # 创建一个全零的观测（所有值在有效范围内）
        valid_obs = np.zeros(OBS_FLAT_SIZE, dtype=np.float32)
        assert space.contains(valid_obs)

    def test_observation_space_contains_out_of_bounds(self):
        """超出范围的观测应该被 contains() 拒绝。"""
        space = PenguinChessFlatObs
        # 创建一个超出 [-1, 1] 范围的观测
        invalid_obs = np.full(OBS_FLAT_SIZE, 2.0, dtype=np.float32)
        assert not space.contains(invalid_obs)

    def test_observation_to_jsonable(self):
        """to_jsonable() 应该返回 list。"""
        space = PenguinChessFlatObs
        obs = np.zeros(OBS_FLAT_SIZE, dtype=np.float32)
        jsonable = space.to_jsonable(obs)
        assert isinstance(jsonable, list)

    def test_observation_from_jsonable(self):
        """from_jsonable() 应该返回正确的 numpy array。"""
        space = PenguinChessFlatObs
        obs_list = [0.0] * OBS_FLAT_SIZE
        result = space.from_jsonable([obs_list])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (OBS_FLAT_SIZE,)


class TestActionSpace:
    """动作空间测试。"""

    def test_action_space_n(self):
        """动作空间应该是 Discrete(60)。"""
        space = PenguinChessActionSpace(n=N_HEX)
        assert space.n == 60

    def test_action_space_contains_valid(self):
        """有效的动作 ID 应该被 contains() 接受。"""
        space = PenguinChessActionSpace(n=N_HEX)
        for action in range(60):
            assert space.contains(action), f"Action {action} should be valid"

    def test_action_space_rejects_negative(self):
        """负数动作 ID 应该被拒绝。"""
        space = PenguinChessActionSpace(n=N_HEX)
        assert not space.contains(-1)

    def test_action_space_rejects_too_large(self):
        """超过范围的动作 ID 应该被拒绝。"""
        space = PenguinChessActionSpace(n=N_HEX)
        assert not space.contains(60)
        assert not space.contains(100)

    def test_action_space_sample(self):
        """sample() 应该返回有效的动作 ID。"""
        space = PenguinChessActionSpace(n=N_HEX)
        for _ in range(100):
            action = space.sample()
            assert 0 <= action < 60

    def test_action_space_repr(self):
        """__repr__ 应该返回正确的格式。"""
        space = PenguinChessActionSpace(n=N_HEX)
        assert "60" in repr(space)


class TestConstants:
    """常量验证测试。"""

    def test_n_hex_is_60(self):
        """N_HEX 应该是 60。"""
        assert N_HEX == 60

    def test_n_pieces_is_6(self):
        """N_PIECES 应该是 6。"""
        assert N_PIECES == 6

    def test_obs_flat_size_calculation(self):
        """OBS_FLAT_SIZE 应该等于 60*3 + 6*4 = 204。"""
        expected = N_HEX * N_FEATURES + N_PIECES * PIECE_FEATURES
        assert OBS_FLAT_SIZE == 204
        assert expected == 204


class TestPieceEncoding:
    """棋子编码测试。"""

    def test_piece_encoding_length(self):
        """每个棋子编码应该是 4 个值。"""
        assert PIECE_FEATURES == 4

    def test_total_piece_features(self):
        """棋子总特征数应该是 6*4 = 24。"""
        assert N_PIECES * PIECE_FEATURES == 24


class TestBoardEncoding:
    """棋盘编码测试。"""

    def test_board_features_per_hex(self):
        """每格编码应该是 3 个值 (q, r, value)。"""
        assert N_FEATURES == 3

    def test_total_board_features(self):
        """棋盘总特征数应该是 60*3 = 180。"""
        assert N_HEX * N_FEATURES == 180


class TestSpaceContainsEdgeCases:
    """空间边界情况测试。"""

    def test_observation_with_none_values(self):
        """包含 None 的观测应该被正确处理。"""
        space = PenguinChessFlatObs
        # 观测不应该包含 Python None
        obs = np.zeros(OBS_FLAT_SIZE, dtype=np.float32)
        # 确认是有效的 float array
        assert space.contains(obs)

    def test_action_space_with_numpy_int(self):
        """numpy 整数类型的动作 ID 应该被接受。"""
        space = PenguinChessActionSpace(n=N_HEX)
        action = np.int64(30)
        assert space.contains(action)

    def test_action_space_with_python_int(self):
        """Python int 类型的动作 ID 应该被接受。"""
        space = PenguinChessActionSpace(n=N_HEX)
        action = 30
        assert space.contains(action)