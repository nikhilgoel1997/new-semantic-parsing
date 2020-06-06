# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


def top_schema_get_vocab(text):
    text = text.replace('[', '')
    text = text.replace(']', '')

    schema_tokens = {'[', ']', 'IN:', 'SL:'}

    for token in text.split(' '):
        if token[:3] in ['IN:', 'SL:']:
            schema_tokens.add(token[3:])
    return schema_tokens
