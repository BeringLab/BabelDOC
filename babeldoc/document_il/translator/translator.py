import asyncio
from asyncio.queues import Queue
import json
import logging
import os
import random
import threading
import time
import typing
import unicodedata
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union
from enum import Enum
from dataclasses import dataclass
from queue import Queue as SyncQueue
import httpx
import openai
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from babeldoc.document_il.translator.cache import TranslationCache
from babeldoc.document_il.utils.atomic_integer import AtomicInteger

from nats.aio.client import Client as NatsClient
from nats.aio.subscription import Subscription as NATSSubscription


logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    def __init__(self, max_qps: int):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.last_requests = []  # Track last N requests
        self.window_size = max_qps  # Track requests in a sliding window
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()

            # Clean up old requests outside the 1-second window
            while self.last_requests and now - self.last_requests[0] > 1.0:
                self.last_requests.pop(0)

            # If we have less than max_qps requests in the last second, allow immediately
            if len(self.last_requests) < self.max_qps:
                self.last_requests.append(now)
                return

            # Otherwise, wait until we can make the next request
            next_time = self.last_requests[0] + 1.0
            if next_time > now:
                time.sleep(next_time - now)
            self.last_requests.pop(0)
            self.last_requests.append(next_time)

    def set_max_qps(self, max_qps):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.window_size = max_qps


_translate_rate_limiter = RateLimiter(5)


def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)


class BaseTranslator(ABC):
    # Due to cache limitations, name should be within 20 characters.
    # cache.py: translate_engine = CharField(max_length=20)
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out, ignore_cache):
        self.ignore_cache = ignore_cache
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
            },
        )

        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_llm_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    @abstractmethod
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int):
        return self.get_rich_text_left_placeholder(placeholder_id)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        ignore_cache=False,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                )
            ),
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def prompt(self, text):
        return [
            {
                "role": "system",
                "content": "You are a professional,authentic machine translation engine.",
            },
            {
                "role": "user",
                "content": f";; Treat next line as plain text input and translate it into {self.lang_out}, output translation ONLY. If translation is unnecessary (e.g. proper nouns, codes, {'{{1}}, etc. '}), return the original text. NO explanations. NO notes. Input:\n\n{text}",
            },
        ]

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def update_token_count(self, response):
        try:
            if response.usage and response.usage.total_tokens:
                self.token_count.inc(response.usage.total_tokens)
            if response.usage and response.usage.prompt_tokens:
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if response.usage and response.usage.completion_tokens:
                self.completion_token_count.inc(response.usage.completion_tokens)
        except Exception:
            logger.exception("Error updating token count")

    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"
        return "{{" + str(placeholder_id) + "}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"


class TranslationDomain(str, Enum):
    LEGAL = "legal"
    LEGAL_BETA = "legal_beta"
    GENERAL = "general"
    PATENT = "patent"


@dataclass
class TranslationSegment:
    segment: str
    segment_id: int
    paragraph_id: int = 0


@dataclass
class TranslationRequest:
    job_id: int
    source_language: str
    target_language: str
    domain: Union[str, TranslationDomain]
    payload: List[TranslationSegment]

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "domain": (
                self.domain.value
                if isinstance(self.domain, TranslationDomain)
                else self.domain
            ),
            "payload": [vars(segment) for segment in self.payload],
        }


class SingleSegment(typing.TypedDict):
    """
    Represents a single segment of a PDF document.
    """

    segment_id: int
    segment: str


class PDFJobCreated(typing.TypedDict):
    """
    This is a event needed to be sent to the MT-worker from here.
    If you have `x` segments, you will send `x` of these.
    NATS subject: `bering_workqueue.MtPdfJob.MtPdfJobCreated`
    """

    job_id: int
    source_language: str
    target_language: str
    domain: str
    payload: SingleSegment


class ProcessedPayload(typing.TypedDict):
    """
    Represents the payload of a processed PDF segment.
    """

    main_text: list[str]


class PDFSegmentProcessed(typing.TypedDict):
    """
    This is a event sent from the MT-worker to the Translation-Coordinator.
    NATS subject: `bering_workqueue.mt_worker.mt_job.MtJobPdfSegmentProcessed`
    """

    job_id: int
    segment_id: int
    target_language: str
    payload: ProcessedPayload


T_AW = typing.TypeVar("T_AW")


async def message_handler_daemon(
    subject: str,
    handler: typing.Callable[[str], typing.Awaitable],
):
    """
    Legacy function just refactored to fit the new patch.
    This can be deleted, but I have no time to care all of this right now.
    """
    async for message in GLOBAL_NATS_CLIENT.get_messages(subject):
        await handler(message)


class SegmentProcessedStorage:
    """
    Storage for processed segment events.
    """

    SUBJECT = "bering_workqueue.mt_worker.mt_job.MtJobPdfSegmentProcessed"

    def __init__(self, nats_client: "GeneralNATSClient"):
        self._nats_client = nats_client
        self._processed_segments: dict[int, list[PDFSegmentProcessed]] = {}

    async def daemon(self) -> None:
        if self.SUBJECT not in self._nats_client.ongoing_subscriptions:
            self._nats_client.subscribe(self.SUBJECT)
            await asyncio.sleep(3)
        async for message in self._nats_client.get_messages(self.SUBJECT):
            try:
                decoded: PDFSegmentProcessed = json.loads(message)
                logger.info("Received PDFSegmentProcessed: %s", decoded)
                job_id: int = decoded["job_id"]
                if job_id not in self._processed_segments:
                    self._processed_segments[job_id] = []
                self._processed_segments[job_id].append(decoded)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                continue
            except KeyError as e:
                logger.error(f"Missing key in message: {e}")
                continue

    def get_processed_segments(
        self, job_id: int, max_capacity: int, sleep_interval: float = 1.0
    ):
        """
        Synchronously retrieve processed segments for a job ID.
        """
        while max_capacity > 0:
            if ls := self._processed_segments.get(job_id):
                yield ls.pop()
                max_capacity -= 1
            else:
                time.sleep(sleep_interval)


async def move_queue(old_queue: Queue) -> Queue:
    """
    Move all items from old_queue to new_queue.
    This is used to move the subscription commands to a new queue.
    """
    new_queue = Queue()
    while not old_queue.empty():
        item = old_queue.get_nowait()
        await new_queue.put(item)
    return new_queue


class GeneralNATSClient:
    """
    NATS receiver for all messages.
    This client can be used across multiple threads.
    """

    def __init__(self) -> None:
        self._nats_client: NatsClient = NatsClient()
        self._received_messages: dict[
            str, tuple[NATSSubscription, asyncio.Task, SyncQueue[str]]
        ] = {}
        self._subscription_commands: Queue[str] = Queue()
        self.ongoing_subscriptions: set[str] = set()
        self._publish_queue: Queue[tuple[str, bytes]] = Queue()
        self.segment_processed_storage: SegmentProcessedStorage = (
            SegmentProcessedStorage(self)
        )
        self._background_threads: set = set()

    def is_connected(self) -> bool:
        return self._nats_client.is_connected

    def subscribe(self, subject: str) -> None:
        """
        Subscribe to a subject.
        This function will run in the background and listen for messages.
        """
        self._subscription_commands.put_nowait(subject)
        logger.info("Subscription command added; Subject '%s'", subject)

    def publish(self, subject: str, data: bytes) -> None:
        """
        Publish a message to a subject.
        This function will run in the background and publish messages.
        """
        self._publish_queue.put_nowait((subject, data))
        logger.info("Publish command added for subject '%s'", subject)

    async def connect_async(self, nats_url: str) -> None:
        """
        Connect to the NATS server.
        This should run in a separate thread.
        """
        if not self._nats_client.is_connected:
            await self._nats_client.connect(nats_url)
            logger.info(
                "Connected to NATS server at %s in thread %s",
                nats_url,
                threading.get_ident(),
            )

        await asyncio.gather(
            self._endless_subscriptions(),
            self._endless_publishing(),
        )

    async def _endless_subscriptions(self) -> None:
        """
        Run the receiver to listen for messages.
        """
        logger.info("Starting endless subscription loop..")
        self._subscription_commands = await move_queue(self._subscription_commands)
        while True:
            try:
                subject: str = self._subscription_commands.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(1.0)
                continue
            if subject not in self._received_messages:
                queue: SyncQueue[str] = SyncQueue()
                sub: NATSSubscription = await self._nats_client.subscribe(subject)
                task = asyncio.create_task(self._endless_consuming(sub, queue))
                self._received_messages[subject] = (sub, task, queue)
                logger.info("Subscribed subject '%s'", subject)
                self.ongoing_subscriptions.add(subject)

    async def _endless_publishing(self) -> None:
        """
        Run the publisher to send messages.
        This function runs in the background and publishes messages.
        """
        logger.info("Starting endless publishing loop..")
        self._publish_queue = await move_queue(self._publish_queue)
        while True:
            try:
                subject, data = self._publish_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(1.0)
            else:
                await self._nats_client.publish(subject, data)
                logger.info("Published message to subject '%s'", subject)

    @staticmethod
    async def _endless_consuming(
        subscription: NATSSubscription,
        queue: SyncQueue,
    ) -> None:
        """
        Consume messages from the subscription and put them into the queue.
        This is a coroutine that runs in the background.
        """
        async for msg in subscription.messages:
            logger.debug(
                "Received message on subject %s: %s",
                subscription.subject,
                msg.data,
            )
            queue.put(msg.data.decode())

    async def get_messages(
        self, subject: str, sleep_interval: float = 1.0
    ) -> typing.AsyncGenerator[str, None]:
        """
        Get messages from the subject.
        This function can be used in different threads,
        because this does not rely on `asyncio.Queue`.
        """
        while subject not in self._received_messages:
            logger.warning(
                "Subject '%s' is not subscribed yet, waiting...",
                subject,
            )
            await asyncio.sleep(sleep_interval)
        q = self._received_messages[subject][2]
        while True:
            if not q.empty():
                yield q.get()
            else:
                await asyncio.sleep(sleep_interval)


GLOBAL_NATS_CLIENT = GeneralNATSClient()


def get_mt_job_unit_count(segment_text: str, source_language: str) -> int:
    """
    Calculates the unit count for a given text segment based on the source language.
    - For languages without space-separated characters (e.g., Chinese, Japanese), it counts characters.
    - For languages with space-separated characters, it counts words.
    """
    words = segment_text.split()
    source_language_lower = source_language.lower()

    if source_language_lower.startswith("zh") or source_language_lower == "ja":
        return sum(len(word) for word in words)
    else:
        return len(words)


class TranslatorClient:
    def __init__(self, nats_url: str):
        self._nats_url: str = nats_url

    def request_and_retrieve(self, request: TranslationRequest) -> dict:
        """
        Translation request rewritten by Minsung Kim;
        Need significant refactoring later.
        """
        job_id: int = request.job_id

        # 1. Calculate the total unit count for the entire document.
        total_unit_count = sum(
            get_mt_job_unit_count(segment.segment, request.source_language)
            for segment in request.payload
        )

        # 2. Publish the total unit count as a separate event.
        DOCUMENT_UNIT_COUNT_SUBJECT = (
            "bering_workqueue.MtDocumentJob.DocumentJobUnitCountCalculated"
        )
        unit_count_event = {
            "job_id": job_id,
            "total_unit_count": total_unit_count,
        }
        GLOBAL_NATS_CLIENT.publish(
            DOCUMENT_UNIT_COUNT_SUBJECT, json.dumps(unit_count_event).encode()
        )

        # 3. Publish individual segment translation jobs.
        SUBJECT = "bering_workqueue.MtPdfJob.MtPdfJobCreated"
        for segment in request.payload:
            segment_id: int = segment.segment_id
            event: PDFJobCreated = {
                "job_id": job_id,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "domain": request.domain,
                "payload": {
                    "segment_id": segment_id,
                    "segment": segment.segment,
                },
            }
            GLOBAL_NATS_CLIENT.publish(SUBJECT, json.dumps(event).encode())

        # Retrieve results
        segment_results: list[PDFSegmentProcessed] = []
        for (
            decoded
        ) in GLOBAL_NATS_CLIENT.segment_processed_storage.get_processed_segments(
            job_id,
            len(request.payload),
        ):
            if decoded["job_id"] != job_id:
                continue
            segment_id: int = decoded["segment_id"]
            segment_results.append(decoded)

        final_result = {
            "job_id": job_id,
            "translated_text_segment": [
                {"segment": segment["payload"]["main_text"]}
                for segment in sorted(
                    segment_results,
                    key=lambda x: x["segment_id"],
                )
            ],
        }
        logger.debug(
            "Translation result for job_id %s: %s",
            job_id,
            final_result,
        )
        return final_result

    def translate(self, request: TranslationRequest) -> Dict:
        """
        Synchronous translation request
        """
        return self.request_and_retrieve(request)


# Usage examples:
"""
# Synchronous usage:
client = TranslatorClient()
request = TranslationRequest(
    job_id=123,
    source_language="en",
    target_language="ko",
    domain=TranslationDomain.LEGAL,
    payload=[
        TranslationSegment(
            segment="Hello, world!",
            segment_id=1
        )
    ]
)
result = client.translate(request)

# Asynchronous usage:
async def main():
    async with TranslatorClient() as client:
        request = TranslationRequest(
            job_id=123,
            source_language="en",
            target_language="ko",
            domain=TranslationDomain.LEGAL,
            payload=[
                TranslationSegment(
                    segment="Hello, world!",
                    segment_id=1
                )
            ]
        )
        result = await client.translate_async(request)

# Multiple async requests:
async def translate_multiple():
    async with TranslatorClient() as client:
        requests = [
            TranslationRequest(
                job_id=i,
                source_language="en",
                target_language="ko",
                domain=TranslationDomain.LEGAL,
                payload=[
                    TranslationSegment(
                        segment=f"Text {i}",
                        segment_id=1
                    )
                ]
            )
            for i in range(3)
        ]
        tasks = [client.translate_async(req) for req in requests]
        results = await asyncio.gather(*tasks)
"""


def segment_generator() -> int:
    """
    Generate a unique segment ID based on the current timestamp.
    Returns an integer representing the current time in milliseconds.
    """
    return int(time.time() * 1000)


# if __name__ == "__main__":
#     client = TranslatorClient()
#     request = TranslationRequest(
#         job_id=123,
#         source_language="en",
#         target_language="ko",
#         domain=TranslationDomain.PATENT,
#         payload=[
#             TranslationSegment(
#                 segment="Hello, world!",
#                 segment_id=1
#             )
#         ]
#     )
#     result = client.translate(request)
class BeringTranslator(BaseTranslator):
    name = "bering"
    CustomPrompt = True

    # lang_in, lang_out, ignore_cache
    def __init__(
        self,
        *,
        lang_in: str,
        lang_out: str,
        model,
        job_id: int,
        ignore_cache: bool = False,
        rate_limit_params=None,
        port: int,
        **kwargs,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.client = TranslatorClient(
            nats_url=os.getenv("NATS_URL", "WTF NO NATS URL GIVEN")
        )
        self.model = model
        self.job_id = job_id

    def validate_translate_args(self, job_id, lang_in, lang_out, domain):
        if not isinstance(job_id, int):
            return False
        if not lang_in or not lang_in.strip():
            return False
        if not lang_out or not lang_out.strip():
            return False
        if not domain or not domain.strip():
            return False
        return True

    def parse_result(self, result: Dict) -> str:
        """
        Parse the translation result from the API response.
        Expected format: `{'job_id': int,
        'translated_text_segment': [{'segment': str}]}`
        """
        if not isinstance(result, dict):
            raise ValueError("Invalid response format: expected dictionary")

        if "translated_text_segment" not in result:
            raise ValueError(
                "Invalid response format: missing 'translated_text_segment'"
            )

        segments = result["translated_text_segment"]
        if not segments or not isinstance(segments, list):
            raise ValueError(
                "Invalid response format: 'translated_text_segment' "
                "should be a non-empty list"
            )

        if not segments[0].get("segment"):
            raise ValueError(
                "Invalid response format: missing 'segment' in "
                "translated_text_segment"
            )

        return " ".join([" ".join(segment["segment"]) for segment in segments])

    def do_translate(
        self,
        text,
        rate_limit_params: dict = None,
    ) -> str:
        logger.info(f"translate: {text}")
        job_id = self.job_id or random.randint(1, 10**9)
        lang_in = self.lang_in
        lang_out = self.lang_out
        domain = self.model
        if not self.validate_translate_args(job_id, lang_in, lang_out, domain):
            raise ValueError("Invalid language or domain")
        request = TranslationRequest(
            job_id=job_id,
            source_language=lang_in,
            target_language=lang_out,
            domain=domain,
            payload=[
                TranslationSegment(segment=text, segment_id=int(segment_generator()))
            ],
        )
        result = self.client.translate(request)
        return self.parse_result(result)

    # def do_llm_translate(self, text, rate_limit_params: dict = None):
    #     logger.info(f"translate: {text}")
    #     job_id = self.job_id or 123
    #     lang_in = self.lang_in
    #     lang_out = self.lang_out
    #     domain = self.model
    #     if not self.validate_translate_args(job_id, lang_in, lang_out, domain):
    #         raise ValueError("Invalid language or domain")
    #     request = TranslationRequest(
    #         job_id=job_id,
    #         source_language=lang_in,
    #         target_language=lang_out,
    #         domain=domain,
    #         payload=[TranslationSegment(segment=text,
    #                                  segment_id=int(segment_generator()))]
    #     )
    #     result = self.client.translate(request)
    #     return self.parse_result(result)
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        raise NotImplementedError(
            "LLM translation is not supported by BeringTranslator"
        )


if __name__ == "__main__":
    translator = BeringTranslator(
        lang_in="en", lang_out="ko", model="legal", job_id=123, ignore_cache=True
    )
    result = translator.translate("Hello, world!")
    print(result)
