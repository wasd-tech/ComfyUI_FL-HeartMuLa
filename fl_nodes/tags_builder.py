"""
FL HeartMuLa Tags Builder Node.
Builds style tags for HeartMuLa music generation.
"""

from typing import Tuple


class FL_HeartMuLa_TagsBuilder:
    """
    Build style tags for HeartMuLa music generation.

    HeartMuLa uses comma-separated tags to control the style of generated music.
    This node provides dropdowns for common options and allows additional custom tags.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "build_tags"
    CATEGORY = "🎵FL HeartMuLa"

    # Genre options
    GENRES = [
        "pop", "rock", "electronic", "jazz", "classical", "hip-hop",
        "r&b", "country", "folk", "metal", "indie", "blues", "reggae",
        "soul", "funk", "disco", "punk", "alternative", "ambient",
        "lo-fi", "acoustic", "orchestral", "cinematic", "edm"
    ]

    # Vocal type options
    VOCAL_TYPES = [
        "female vocal", "male vocal", "duet", "choir", "instrumental",
        "vocal harmony", "rap", "spoken word"
    ]

    # Mood options
    MOODS = [
        "energetic", "melancholic", "uplifting", "calm", "aggressive",
        "romantic", "dreamy", "dark", "happy", "sad", "nostalgic",
        "epic", "peaceful", "intense", "playful", "mysterious"
    ]

    # Tempo options
    TEMPOS = ["slow", "medium", "fast", "very slow", "very fast"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genre": (
                    cls.GENRES,
                    {
                        "default": "pop",
                        "tooltip": "Primary music genre"
                    }
                ),
                "vocal_type": (
                    cls.VOCAL_TYPES,
                    {
                        "default": "female vocal",
                        "tooltip": "Type of vocals"
                    }
                ),
                "mood": (
                    cls.MOODS,
                    {
                        "default": "energetic",
                        "tooltip": "Overall mood/feeling"
                    }
                ),
            },
            "optional": {
                "tempo": (
                    cls.TEMPOS,
                    {
                        "default": "medium",
                        "tooltip": "Song tempo"
                    }
                ),
                "secondary_genre": (
                    ["none"] + cls.GENRES,
                    {
                        "default": "none",
                        "tooltip": "Optional secondary genre for fusion styles"
                    }
                ),
                "instruments": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Specific instruments (e.g., piano,guitar,drums,synth)"
                    }
                ),
                "additional_tags": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Any additional style tags (comma-separated)"
                    }
                ),
            }
        }

    def build_tags(
        self,
        genre: str,
        vocal_type: str,
        mood: str,
        tempo: str = "medium",
        secondary_genre: str = "none",
        instruments: str = "",
        additional_tags: str = ""
    ) -> Tuple[str]:
        """
        Build comma-separated style tags.

        Args:
            genre: Primary music genre
            vocal_type: Type of vocals
            mood: Overall mood
            tempo: Song tempo
            secondary_genre: Optional secondary genre
            instruments: Comma-separated instruments
            additional_tags: Any additional tags

        Returns:
            Tuple containing comma-separated tags string
        """
        tags = []

        # Add primary genre
        tags.append(genre)

        # Add secondary genre if specified
        if secondary_genre and secondary_genre != "none":
            tags.append(secondary_genre)

        # Add vocal type
        tags.append(vocal_type)

        # Add mood
        tags.append(mood)

        # Add tempo
        tags.append(tempo)

        # Add instruments if specified
        if instruments.strip():
            # Split and clean instrument list
            for inst in instruments.split(","):
                inst = inst.strip().lower()
                if inst:
                    tags.append(inst)

        # Add additional tags if specified
        if additional_tags.strip():
            for tag in additional_tags.split(","):
                tag = tag.strip().lower()
                if tag:
                    tags.append(tag)

        # Join all tags
        tags_string = ",".join(tags)

        print(f"[FL HeartMuLa] Built tags: {tags_string}")

        return (tags_string,)
