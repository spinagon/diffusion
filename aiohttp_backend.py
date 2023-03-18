import json

import aiohttp


class Http_backend:
    @classmethod
    async def init(cls):
        self = cls()
        self.session = aiohttp.ClientSession()
        return self

    async def get(self, *args, **kwargs):
        async with self.session.get(*args, **kwargs) as r:
            return await Http_response.init(r)

    async def post(self, *args, **kwargs):
        async with self.session.post(*args, **kwargs) as r:
            return await Http_response.init(r)

    async def close(self):
        await self.session.close()


class Http_response:
    @classmethod
    async def init(cls, resp):
        self = cls()
        self.status_code = resp.status
        self.content = await resp.read()
        return self

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content)
