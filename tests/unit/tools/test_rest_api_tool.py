import pytest


class TestRestApi:
    @pytest.fixture()
    def client(self, mocker):
        from griptape.tools import RestApiTool

        mock_return_value = {"value": "foo bar"}

        mock_response = mocker.Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_return_value
        mocker.patch("requests.put", return_value=mock_response)

        mock_response = mocker.Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_return_value
        mocker.patch("requests.post", return_value=mock_response)

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_return_value
        mocker.patch("requests.get", return_value=mock_response)

        mock_response = mocker.Mock()
        mock_response.status_code = 204
        mock_response.json.return_value = mock_return_value
        mocker.patch("requests.delete", return_value=mock_response)

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_return_value
        mocker.patch("requests.patch", return_value=mock_response)

        return RestApiTool(base_url="http://www.griptape.ai", description="Griptape website.")

    def test_put(self, client):
        response = client.put({"values": {"body": {}}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 201

    def test_post(self, client):
        response = client.post({"values": {"body": {}}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 201

    def test_get_one(self, client):
        response = client.get({"values": {"path_params": ["1"]}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 200

    def test_get_all(self, client):
        response = client.get({"values": {}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 200

    def test_get_filtered(self, client):
        response = client.get({"values": {"query_params": {"limit": 10}}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 200

    def test_delete_one(self, client):
        response = client.delete({"values": {"path_params": ["1"]}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 204

    def test_delete_multiple(self, client):
        response = client.delete({"values": {"query_params": {"ids": [1, 2]}}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 204

    def test_patch(self, client):
        response = client.patch({"values": {"path_params": ["1"], "body": {}}})
        assert response.value == {"value": "foo bar"}
        assert response.meta["status_code"] == 200

    def test_build_url(self, client):
        url = client._build_url("https://foo.bar")

        assert url == "https://foo.bar"

        url = client._build_url("https://foo.bar", path="/foo")

        assert url == "https://foo.bar/foo"

        url = client._build_url("https://foo.bar", path_params=[1, "fizz"])

        assert url == "https://foo.bar/1/fizz"

        url = client._build_url("https://foo.bar", path="foo/", path_params=["fizz", "buzz", 1, 2, 3])

        assert url == "https://foo.bar/foo/fizz/buzz/1/2/3"

        url = client._build_url("https://foo.bar", path_params=["fizz", "buzz", 1, 2, 3])

        assert url == "https://foo.bar/fizz/buzz/1/2/3"
