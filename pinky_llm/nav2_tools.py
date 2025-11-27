from langchain.tools import tool
from typing import List, Dict

def make_nav_tools(nav):
    @tool
    def where_am_i() -> str:
        """
        현재 핑키의 위치를 알려줍니다.
        """
        return nav.where_am_i()

    @tool
    def go_to(place: str) -> str:
        """
        (입구, 복도, 거실, 안방) 중 원하는 장소로 핑키가 이동합니다.
        """
        return nav.go_to(place)

    @tool
    def turn_around() -> str:
        """
        핑키가 제자리에서 한바퀴를 돕니다.
        """
        return nav.turn_around(-1.57, 4)

    @tool
    def list_places() -> str:
        """
        핑키가 주행할 수 있는 장소 목록을 보여줍니다.
        장소 목록은 (입구, 복도, 거실, 안방) 중 하나입니다.
        """
        return ", ".join(nav.list_places())
    
    @tool
    def go_forward_a_little() -> str:
        """
        핑키가 조금 전진 주행합니다.
        """
        return nav.go_forward_a_little(0.1, 1.0)

    @tool
    def go_backward_a_little() -> str:
        """
        핑키가 조금 후진 주행합니다.
        """
        return nav.go_backward_a_little(-0.1, 1.0)
    
    @tool
    def turn_left_a_little() -> str:
        """
        핑키가 조금만 왼쪽으로 회전합니다.
        """
        return nav.turn_left_a_little(0.524, 1.0)

    @tool
    def turn_right_a_little() -> str:
        """
        핑키가 조금만 오른쪽으로 회전합니다.
        """
        return nav.turn_right_a_little(-0.524, 1.0)
    
    @tool
    def lets_dance() -> str:
        """
        핑키가 좌우로 흔들며 춤을 춥니다.
        """
        return nav.lets_dance()
    
    @tool
    def what_time_is_it_now() -> str:
        """
        핑키가 현재 시각을 반환합니다. 
        13~24시 일 경우, 12시를 빼고 반환 합니다.
        예)16시04분 -> 4시 4분
        """
        return nav.what_time_is_it_now()
    
    @tool
    def what_date_today() -> str:
        """
        핑키가 현재 날짜와 요일을 반환합니다. 
        날짜만 필요한 경우, 달과 날짜만 반환합니다.
        요일만 필요한 경우, 요일만 반환합니다.
        """
        return nav.what_date_today()

    @tool
    def whats_the_weather(city: str = "Busan") -> str:
        """
        핑키가 지역에 따른 오늘 날씨를 반환합니다.
        키워드에 지역이 따로 없다면 Busan 날씨를 알려줍니다.
        검색할 수 있는 지역: Busan, Seoul, Incheon, Daegu, Gwangju, Daejeon, Ulsan, Jeju
        """
        return nav.whats_the_weather(city)
    
    @tool
    def summarizing_news(search_keyword: str) -> List[Dict[str, str]]:
        """
        핑키가 질문의 키워드로 뉴스 기사 검색을 합니다.
        Web 기반 정보가 필요할 때 사용해도 좋습니다.
        """
        return nav.summarizing_news(search_keyword)
    
    @tool
    def watching_at() -> str:
        """
        핑키 카메라 시점에서 보이는 객체들을 반환합니다.
        """
        return nav.get_detect_info()

    return [
        where_am_i, 
        go_to, 
        turn_around, 
        list_places, 
        go_forward_a_little, 
        go_backward_a_little, 
        turn_left_a_little, 
        turn_right_a_little, 
        lets_dance,
        what_time_is_it_now,
        what_date_today,
        whats_the_weather,
        summarizing_news,
        watching_at,
    ]
